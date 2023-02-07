import wandb
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment

from utils.tools import *


class PropertyPredictor:
    def __init__(self, ocr, config: dict, dataset_config) -> None:
        super().__init__()
        self._property_list = dataset_config.property_order_in_state
        self._matching_mode = config.matching_mode
        (
            self._target_prop_indices,
            self._output_prop_indices,
        ) = self._get_property_indices(dataset_config.properties)
        output_size = self._output_size = self._output_prop_indices[-1][1]
        if ocr.name in ["SLATE", "SlotAttn", "Iodine"]:
            input_size = self._input_size = ocr.rep_dim
            self._use_slot = True
        elif ocr.name in ["VAE"]:
            self._num_slots_for_dist_rep = config.num_slots_for_dist_rep
            input_size = self._input_size = ocr.rep_dim
            output_size = self._output_size = output_size * self._num_slots_for_dist_rep
            self._use_slot = False
        else:
            raise ValueError(f"{ocr.name} is not supported to predict property.")
        self._encoder = ocr
        if config.model_type == "linear":
            self._module = nn.Sequential(nn.Linear(input_size, output_size))
        elif config.model_type == "mlp3":
            hidden_size = 256
            self._module = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, output_size),
            )

        self._ce = nn.CrossEntropyLoss()  # for color, shape and size
        self._mse = nn.MSELoss()  # for x and y coordinates

        # optimizer
        self._opt = torch.optim.Adam(self._module.parameters(), lr=config.learning.lr)

    def _get_property_indices(self, config):
        target_indices = []
        output_indices = []
        target_prev_idx = 0
        output_prev_idx = 0
        for prop_name in self._property_list:
            prop_config = config[prop_name]
            if prop_name == "xy":
                target_indices.append([target_prev_idx, target_prev_idx + 2])
                output_indices.append(
                    [output_prev_idx, output_prev_idx + prop_config.dims]
                )
            else:
                target_indices.append([target_prev_idx, target_prev_idx + 1])
                output_indices.append(
                    [output_prev_idx, output_prev_idx + prop_config.num_candidates]
                )
            target_prev_idx = target_indices[-1][1]
            output_prev_idx = output_indices[-1][1]
        return target_indices, output_indices

    def wandb_watch(self, config):
        wandb.watch(self._module, log=config.log)
        # self._encoder.wandb_watch(config)

    def _get_target_output_for_given_prop_idx(self, idx, target, output):
        st_idx, nd_idx = self._target_prop_indices[idx]
        res_target = target[:, st_idx:nd_idx]
        st_idx, nd_idx = self._output_prop_indices[idx]
        res_output = output[:, st_idx:nd_idx]
        return res_target, res_output

    def _get_loss_matrix(self, target, output, loss="ce"):
        num_objs, num_slots = target.shape[0], output.shape[0]
        loss_matrix = torch.zeros((num_objs, num_slots)).to(target.device)
        for o_idx in range(num_objs):
            for s_idx in range(num_slots):
                if loss == "ce":
                    loss_matrix[o_idx, s_idx] = self._ce(output[s_idx:s_idx+1], target[o_idx:o_idx+1])
                elif loss == "mse":
                    loss_matrix[o_idx, s_idx] = self._mse(output[s_idx], target[o_idx])
                else:
                    raise ValueError(f"{loss} must be CE or MSE.")
        return loss_matrix

    def get_loss(self, batch: List) -> dict:
        x = batch["obss"]
        y = batch["objs"]
        # feed forward
        x = self._encoder(x).detach()
        if self._use_slot:
            # [batch size, num of slots, dimension]
            B, N, D = x.shape
            output = self._module(x.reshape(B * N, D)).reshape(B, N, self._output_size)
        else:
            # [batch size, dimension]
            B, D = x.shape
            # [batch size, num of slots, dimension]
            output = self._module(x).reshape(
                B,
                self._num_slots_for_dist_rep,
                self._output_size // self._num_slots_for_dist_rep,
            )
        # get loss matrix after alignment [B, num_objs, num_objs]
        B, num_objs, _ = y.shape
        B, num_slots, _ = output.shape
        num_props = len(self._property_list)
        col_inds, aligned_loss_matrix = [], []
        for b_idx in range(B):
            loss_matrix = []
            for idx in range(num_props):
                _target, _output = self._get_target_output_for_given_prop_idx(
                    idx, y[b_idx], output[b_idx]
                )
                if self._property_list[idx] == "xy":
                    loss_matrix.append(self._get_loss_matrix(_target, _output, "mse"))
                else:
                    loss_matrix.append(
                        self._get_loss_matrix(
                            _target.squeeze(-1).to(y.device, dtype=torch.int64),
                            F.softmax(_output, dim=-1),
                            "ce",
                        )
                    )
            loss_matrix = torch.sum(torch.stack(loss_matrix, 0), 0)
            _, col_ind = linear_sum_assignment(loss_matrix.cpu().detach().numpy())
            col_ind = torch.LongTensor(col_ind).to(y.device)
            col_inds.append(col_ind)
            aligned_loss_matrix.append(torch.index_select(loss_matrix, 1, col_ind))

        metrics = {}
        # get loss
        metrics["loss"] = torch.sum(
            torch.diagonal(torch.sum(torch.stack(aligned_loss_matrix, 0), 0))
        )
        # get accuracy and R^2
        for idx in range(num_props):
            if self._property_list[idx] == "xy":
                metrics[f"R^2_{self._property_list[idx]}"] = []
                metrics[f"mse_{self._property_list[idx]}"] = []
            else:
                metrics[f"acc_{self._property_list[idx]}"] = []
        B, num_objs, _ = y.shape
        for b_idx in range(B):
            _output = torch.index_select(output[b_idx], 0, col_inds[b_idx])
            for idx in range(num_props):
                _target, __output = self._get_target_output_for_given_prop_idx(
                    idx, y[b_idx], _output
                )
                if self._property_list[idx] == "xy":
                    m_name = f"R^2_{self._property_list[idx]}"
                    if not m_name in metrics.keys():
                        metrics[m_name] = []
                    _target_mean = torch.mean(_target, dim=0)
                    sst = torch.norm(_target - _target_mean.unsqueeze(0), p=2, dim=0)**2
                    sse = torch.norm(__output - _target_mean.unsqueeze(0), p=2, dim=0)**2
                    metrics[m_name].append(sse/sst)
                    m_name = f"mse_{self._property_list[idx]}"
                    metrics[m_name].append(torch.norm(__output - _target, p=2, dim=-1))
                else:
                    m_name = f"acc_{self._property_list[idx]}"
                    if not m_name in metrics.keys():
                        metrics[m_name] = []
                    metrics[m_name].append(
                        torch.argmax(__output, dim=-1) == _target.squeeze(-1)
                    )
        for m_name in metrics.keys():
            if m_name == "loss":
                continue
            else:
                metrics[m_name] = torch.cat(metrics[m_name], dim=0)
                metrics[m_name] = torch.sum(metrics[m_name]) / metrics[m_name].shape[0]
        return metrics

    def train(self) -> None:
        self._module.train()
        self._encoder.train()
        return None

    def eval(self) -> None:
        self._module.eval()
        self._encoder.eval()
        return None

    def to(self, device: str) -> None:
        self._module.to(device)
        self._encoder.to(device)

    def get_samples(self, obs: Tensor) -> dict:
        return self._encoder.get_samples(obs)

    def update(self, batch: List, step: int) -> dict:
        metrics = self.get_loss(batch)
        self._opt.zero_grad()
        self._encoder.set_zero_grad()
        metrics["loss"].backward()
        self._opt.step()
        self._encoder.do_step()
        return metrics

    def save(self) -> dict:
        checkpoint = {}
        checkpoint["property_predictor_module_state_dict"] = self._module.state_dict()
        checkpoint["property_predictor_opt_state_dict"] = self._opt.state_dict()
        checkpoint.update(self._encoder.save())
        return checkpoint

    def load(self, checkpoint: str) -> None:
        self._module.load_state_dict(checkpoint["property_predictor_module_state_dict"])
        self._opt.load_state_dict(checkpoint["property_predictor_opt_state_dict"])
        self._encoder.load(checkpoint)
