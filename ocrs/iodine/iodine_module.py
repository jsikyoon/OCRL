# Modified from https://github.com/singhgautam/whatwhere-slate/blob/gautam/iodine-005-clean/clevrmetal-001/iodine/iodine.py
import numpy as np
import torch
from torch import nn
from torch.autograd import grad
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F

from ocrs.common.utils import visualize
from utils.tools import Tensor, for_viz, calculate_ari


class Iodine_Module(nn.Module):
    def __init__(self, ocr_config: dict, env_config: dict):
        nn.Module.__init__(self)
        self.slot_size = ocr_config.slot_size
        self.num_iterations = ocr_config.num_iterations
        self.num_slots = ocr_config.num_slots
        self.img_channels = ocr_config.img_channels
        self.img_size = env_config.obs_size
        self.beta = ocr_config.beta
        self.sigma = ocr_config.sigma
        self.use_layernorm = ocr_config.layer_norm
        self.rep_dim = self.slot_size

        self.encodings = [
            "posterior",
            "grad_post",
            "image",
            "means",
            "mask",
            "mask_logits",
            "mask_posterior",
            "grad_means",
            "grad_mask",
            "likelihood",
            "leave_one_out_likelihood",
            "coordinate",
        ]

        # architecture
        input_size, lambda_size = self.get_input_size()

        self.refine = RefinementNetwork(
            input_size,
            ocr_config.ref_cnn_hidden_size,
            ocr_config.ref_mlp_hidden_size,
            ocr_config.slot_size,
            ocr_config.ref_cnn_layers,
            kernel_size=ocr_config.ref_cnn_kernel_size,
            stride=ocr_config.ref_cnn_stride_size,
        )

        self.decoder = Decoder(
            dim_in=ocr_config.slot_size,
            dim_hidden=ocr_config.dec_cnn_hidden_size,
            n_layers=ocr_config.dec_cnn_layers,
            kernel_size=ocr_config.dec_cnn_kernel_size,
            img_size=self.img_size,
        )

        self.slot_mean_init = nn.Parameter(
            torch.zeros(1, 1, self.slot_size), requires_grad=True
        )
        self.slot_logsig_init = nn.Parameter(
            torch.zeros(1, 1, self.slot_size), requires_grad=True
        )
        #self.slot_mean_init = nn.Parameter(
        #    torch.randn(1, 1, self.slot_size), requires_grad=True
        #)
        #self.slot_logsig_init = nn.Parameter(
        #    torch.randn(1, 1, self.slot_size), requires_grad=True
        #)
        self.slot_init = nn.Parameter(
            torch.zeros(1, 1, self.slot_size), requires_grad=True
        )

    @torch.enable_grad()
    def _forward(self, image):
        """
        :param image: (B, 3, H, W)
        :return: loss
        """
        B, C, H, W = image.size()
        elbos = []

        slot_means = self.slot_mean_init.repeat(B, self.num_slots, 1)
        slot_logsigs = self.slot_logsig_init.repeat(B, self.num_slots, 1)
        lstm_hidden = None

        for i in range(self.num_iterations):
            # Compute ELBO
            post_dist = Normal(slot_means, slot_logsigs.exp())
            slots = post_dist.rsample()  # (B, K, L)

            recons, mask_logits = self.decoder(
                slots
            )  # (B, K, 3, H, W), (B, K, 1, H, W)

            masks = F.softmax(mask_logits, dim=1)  # (B, K, 1, H, W)

            recons_masked = masks * recons  # (B, K, C, H, W)
            recon = torch.sum(recons_masked, dim=1)  # (B, C, H, W)
            mse = ((image - recon) ** 2).sum() / B

            #if i == self.num_iterations - 1:
            #    import ipdb
            #    ipdb.set_trace()

            kl = kl_divergence(post_dist, Normal(0, 1))  # B
            kl = kl.sum() / B  # 1

            component_log_probs = Normal(recons, self.sigma).log_prob(
                image.unsqueeze(1)
            )  # B, K, 3, H, W
            pixel_log_likelihood = torch.logsumexp(
                (masks + 1e-12).log() + component_log_probs, dim=1, keepdim=True
            )  # B, 1, 3, H, W
            log_likelihood = pixel_log_likelihood.sum() / B  # 1

            elbo = log_likelihood - (self.beta * kl)
            elbos.append(elbo)

            # Refine
            if i < self.num_iterations - 1:
                # Build auxiliary inputs
                encoding = image.new_zeros(0)
                latent = image.new_zeros(0)

                # gradients
                slot_means_grad, slot_logsigs_grad, recons_grad, masks_grad = grad(
                    B * elbo,
                    [slot_means, slot_logsigs, recons, masks],
                    create_graph=self.training,
                    retain_graph=self.training,
                )  # (B, K, L)
                slot_means_grad, slot_logsigs_grad, recons_grad, masks_grad = (
                    slot_means_grad.detach(),
                    slot_logsigs_grad.detach(),
                    recons_grad.detach(),
                    masks_grad.detach(),
                )

                if "posterior" in self.encodings:
                    latent = torch.cat((latent, slot_means, slot_logsigs), dim=-1)

                if "grad_post" in self.encodings:
                    if self.use_layernorm:
                        slot_means_grad = self.layernorm(slot_means_grad)
                        slot_logsigs_grad = self.layernorm(slot_logsigs_grad)

                    latent = torch.cat(
                        [latent, slot_means_grad, slot_logsigs_grad], dim=-1
                    )

                if "image" in self.encodings:
                    encoding = torch.cat(
                        [encoding, image[:, None].repeat(1, self.num_slots, 1, 1, 1)],
                        dim=2,
                    )

                if "means" in self.encodings:
                    encoding = torch.cat([encoding, recons], dim=2)

                if "mask" in self.encodings:
                    encoding = torch.cat([encoding, masks], dim=2)

                if "mask_logits" in self.encodings:
                    encoding = torch.cat([encoding, mask_logits], dim=2)

                if "mask_posterior" in self.encodings:
                    a_component_log_probs = component_log_probs.sum(
                        dim=2, keepdim=True
                    )  # B, K, 1, H, W
                    a_component_probs = torch.log_softmax(
                        a_component_log_probs, dim=1
                    )  # B, K, 1, H, W
                    encoding = torch.cat([encoding, a_component_probs], dim=2)

                if "grad_means" in self.encodings:
                    if self.use_layernorm:
                        recons_grad = self.layernorm(recons_grad)
                    encoding = torch.cat([encoding, recons_grad], dim=2)

                if "grad_mask" in self.encodings:
                    if self.use_layernorm:
                        masks_grad = self.layernorm(masks_grad)
                    encoding = torch.cat([encoding, masks_grad], dim=2)

                if "likelihood" in self.encodings:
                    a_pixel_log_likelihood = pixel_log_likelihood.sum(
                        dim=2, keepdim=True
                    ).exp()  # (B, 1, 1, H, W)
                    a_pixel_log_likelihood = a_pixel_log_likelihood.repeat(
                        1, self.num_slots, 1, 1, 1
                    )  # (B, K, 1, H, W)
                    if self.use_layernorm:
                        a_pixel_log_likelihood = self.layernorm(a_pixel_log_likelihood)
                    encoding = torch.cat(
                        [encoding, a_pixel_log_likelihood.detach()], dim=2
                    )

                if "leave_one_out_likelihood" in self.encodings:
                    a_component_log_probs = component_log_probs.sum(
                        dim=2, keepdim=True
                    )  # B, K, 1, H, W
                    a_component_probs = a_component_log_probs.exp()  # B, K, 1, H, W
                    a_component_probs_sum = (masks * a_component_probs).sum(
                        dim=1, keepdim=True
                    )  # B, 1, 1, H, W
                    leave_one_out = a_component_probs_sum - (
                        masks * a_component_probs
                    )  # B, K, 1, H, W
                    leave_one_out = leave_one_out / (1 - masks + 1e-5)  # B, K, 1, H, W
                    if self.use_layernorm:
                        leave_one_out = self.layernorm(leave_one_out)  # B, K, 1, H, W
                    encoding = torch.cat([encoding, leave_one_out.detach()], dim=2)

                if "coordinate" in self.encodings:
                    xx = torch.linspace(-1, 1, W, device=image.device)
                    yy = torch.linspace(-1, 1, H, device=image.device)
                    yy, xx = torch.meshgrid((yy, xx))
                    # (2, H, W)
                    coords = torch.stack((xx, yy), dim=0)
                    coords = (
                        coords[None, None].repeat(B, self.num_slots, 1, 1, 1).detach()
                    )
                    encoding = torch.cat([encoding, coords], dim=2)

                # Apply refinement
                mean_delta, logsig_delta, lstm_hidden = self.refine(
                    encoding, latent, lstm_hidden
                )
                slot_means = slot_means + mean_delta
                slot_logsigs = slot_logsigs + logsig_delta

        elbo = 0
        for i, e in enumerate(elbos):
            elbo = elbo + (i + 1) / len(elbos) * e

        return (
            slots,
            recon.clamp(0.0, 1.0),
            recons_masked.clamp(0.0, 1.0),
            masks,
            -elbo,
            mse,
            kl,
            recons.clamp(0.0, 1.0),
            masks,
        )

    def forward(self, obs: Tensor, with_masks) -> Tensor:
        slots, recon, recons_masked, masks, loss, mse, kl, _, _ = self._forward(obs)
        if with_masks:
            return slots, masks
        else:
            return slots

    def get_loss(self, obs: Tensor, masks: Tensor, with_rep=False) -> Tensor:
        _, _, _, attns, loss, mse, kl, _, _ = self._forward(obs)
        fg_mask = (1 - masks[:,-1].unsqueeze(1))
        attns = attns * fg_mask
        attns = torch.cat([attns, fg_mask], dim=1)
        ari = np.mean(calculate_ari(masks, attns))
        metrics = {"loss": loss, "mse": mse.detach(), "ari": ari, "kld": kl.detach()}
        return metrics

    def get_samples(self, obs: Tensor) -> dict:
        slots, recon, recons_masked, masks, loss, mse, kl, means, masks = self._forward(obs)
        return {"samples": for_viz(visualize([obs, recon, recons_masked, masks.repeat(1, 1, 3, 1, 1), means]))}
        #return {"samples": for_viz(visualize([obs, recon]))}
        # return {"samples": np.concatenate([for_viz(obs), for_viz(recon), for_viz(recons_masked)], axis=-2)}

    def get_input_size(self):
        size = 0
        latent = 0

        if "grad_post" in self.encodings:
            latent += 2 * self.slot_size
        if "posterior" in self.encodings:
            latent += 2 * self.slot_size
        if "image" in self.encodings:
            size += self.img_channels
        if "means" in self.encodings:
            size += self.img_channels
        if "mask" in self.encodings:
            size += 1
        if "mask_logits" in self.encodings:
            size += 1
        if "mask_posterior" in self.encodings:
            size += 1
        if "grad_means" in self.encodings:
            size += self.img_channels
        if "grad_mask" in self.encodings:
            size += 1
        if "likelihood" in self.encodings:
            size += 1
        if "leave_one_out_likelihood" in self.encodings:
            size += 1
        if "coordinate" in self.encodings:
            size += 2

        return size, latent

    @staticmethod
    def layernorm(x):
        """
        :param x: (B, K, L) or (B, K, C, H, W)
        :return:
        """
        if len(x.size()) == 3:
            layer_mean = x.mean(dim=2, keepdim=True)
            layer_std = x.std(dim=2, keepdim=True)
        elif len(x.size()) == 5:
            mean = (
                lambda x: x.mean(2, keepdim=True)
                .mean(3, keepdim=True)
                .mean(4, keepdim=True)
            )
            layer_mean = mean(x)
            # this is not implemented in some version of torch
            layer_std = torch.pow(x - layer_mean, 2)
            layer_std = torch.sqrt(mean(layer_std))
        else:
            assert False, "invalid size for layernorm"

        x = (x - layer_mean) / (layer_std + 1e-5)
        return x


class Decoder(nn.Module):
    """
    Given sampled latent variable, output RGB+mask
    """

    def __init__(self, dim_in, dim_hidden, n_layers, kernel_size, img_size):
        nn.Module.__init__(self)

        padding = kernel_size // 2
        self.broadcast = SpatialBroadcast()
        self.mlc = MultiLayerConv(dim_in + 2, dim_hidden, n_layers, kernel_size)
        self.conv = nn.Conv2d(
            dim_hidden, 4, kernel_size=kernel_size, stride=1, padding=padding
        )
        self.img_size = img_size

    def forward(self, x):
        """
        :param x: (B, K, N, H, W), where N is the number of latent dimensions
        :return: (B, K, 3, H, W) (B, K, 1, H, W), where 4 is RGB + mask
        """
        B, K, *ORI = x.size()
        x = x.view(B * K, *ORI)

        x = self.broadcast(x, self.img_size, self.img_size)
        x = self.mlc(x)
        x = self.conv(x)

        mean, mask = torch.split(x, [3, 1], dim=1)
        #mean = torch.sigmoid(mean)

        BK, *ORI = mean.size()
        mean = mean.view(B, K, *ORI)
        BK, *ORI = mask.size()
        mask = mask.view(B, K, *ORI)
        return mean, mask


class RefinementNetwork(nn.Module):
    """
    Given input encoding, output updates to lambda.
    """

    def __init__(
        self, dim_in, dim_conv, dim_hidden, dim_out, n_layers, kernel_size, stride
    ):
        """
        :param dim_in: input channels
        :param dim_conv: conv output channels
        :param dim_hidden: MLP and LSTM output dim
        :param dim_out: latent variable dimension
        """
        nn.Module.__init__(self)

        self.mlc = MultiLayerConv(
            dim_in, dim_conv, n_layers, kernel_size, stride=stride
        )
        self.mlp = MLP(dim_conv, dim_hidden, n_layers=1)

        self.lstm = nn.LSTMCell(dim_hidden + 4 * dim_out, dim_hidden)
        self.mean_update = nn.Linear(dim_hidden, dim_out)
        self.logsig_update = nn.Linear(dim_hidden, dim_out)

    def forward(self, x, latent, hidden=(None, None)):
        """
        :param x: (B, K, D, H, W), where D varies for different input encodings
        :param latent: (B, K, L * 4), contains posterior parameters and gradients
        :param hidden: a tuple (c, h)
        :return: (B, K, L), (B, K, L), (h, c) for mean and gaussian respectively, where L is
                 the latent dimension. And hidden state
        """
        B, K, *ORI = x.size()
        x = x.view(B * K, *ORI)
        B, K, *ORI = latent.size()
        latent = latent.view(B * K, *ORI)

        # (BK, D, H, W)
        x = self.mlc(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # (BK, D)
        x = x.view(B * K, -1)
        # to uniform length
        x = F.elu(self.mlp(x))
        # concatenate
        x = torch.cat((x, latent), dim=1)
        (c, h) = self.lstm(x, hidden)

        # update
        mean_delta = self.mean_update(h)
        logsig_delta = self.logsig_update(h)

        BK, *ORI = mean_delta.size()
        mean_delta = mean_delta.view(B, K, *ORI)
        BK, *ORI = logsig_delta.size()
        logsig_delta = logsig_delta.view(B, K, *ORI)

        return mean_delta, logsig_delta, (c, h)


class SpatialBroadcast(nn.Module):
    """
    Broadcast a 1-D variable to 3-D, plus two coordinate dimensions
    """

    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, x, width, height):
        """
        :param x: (B, L)
        :param coordinate: whether to a the coordinate dimension
        :return: (B, L + 2, W, H)
        """
        # B, K, *ORI = x.size()
        # x = x.view(B*K, *ORI)

        B, L = x.size()
        # (B, L, 1, 1)
        x = x[:, :, None, None]
        # (B, L, W, H)
        x = x.repeat(1, 1, width, height)
        xx = torch.linspace(-1, 1, width, device=x.device)
        yy = torch.linspace(-1, 1, height, device=x.device)
        yy, xx = torch.meshgrid((yy, xx))
        # (2, H, W)
        coords = torch.stack((xx, yy), dim=0)
        # (B, 2, H, W)
        coords = coords[None].repeat(B, 1, 1, 1)

        # (B, L + 2, W, H)
        x = torch.cat((x, coords), dim=1)

        # BK, *ORI = x.size()
        # x = x.view(B, K, *ORI)

        return x


class MLP(nn.Module):
    """
    Multi-layer perception using elu
    """

    def __init__(self, dim_in, dim_out, n_layers):
        nn.Module.__init__(self)
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.layers = nn.ModuleList([])

        for i in range(n_layers):
            self.layers.append(nn.Linear(dim_in, dim_out))
            dim_in = dim_out

    def forward(self, x):
        """
        :param x: (B, Din)
        :return: (B, Dout)
        """
        for layer in self.layers:
            x = F.elu(layer(x))

        return x


class MultiLayerConv(nn.Module):
    """
    Multi-layer convolutional layer
    """

    def __init__(self, dim_in, dim_out, n_layers, kernel_size, stride=1):
        nn.Module.__init__(self)
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.layers = nn.ModuleList([])
        padding = kernel_size // 2

        for i in range(n_layers):
            self.layers.append(
                nn.Conv2d(
                    dim_in,
                    dim_out,
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                )
            )
            dim_in = dim_out

    def forward(self, x):
        """
        :param x: (B, C, H, W)
        :return: (B, C, H, W)
        """
        for layer in self.layers:
            x = F.elu(layer(x))

        return x
