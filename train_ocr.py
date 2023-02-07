import logging
import pathlib
from contextlib import nullcontext

import hydra
import numpy as np
import torch
import tqdm
import wandb

import ocrs
import utils
from utils.tools import *

log = logging.getLogger(__name__)


@hydra.main(config_path="configs/", config_name="train_ocr")
def main(config):
    log_name = get_log_prefix(config)
    log_name += f"-{config.dataset.name}"
    tags = config.tags.split(",") + config.dataset.tags.split(",")
    init_wandb(config, "TrainOCR-" + log_name, tags=tags)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Data Loader
    train_dl, val_dl = get_dataloaders(
        config.dataset,
        config.batch_size,
        config.num_workers
    )

    # Model
    if config.ocr.name == "MAE":
        config.ocr.learning.lr = config.ocr.learning.lr * config.batch_size/256
    model = getattr(ocrs, config.ocr.name)(config.ocr, config.dataset)
    model.wandb_watch(config.wandb)
    model.to(config.device)

    # load
    step, epoch, best_val_loss = load(model,
            resume_checkpoint=config.load.resume_checkpoint,
            resume_run_path=config.load.resume_run_path)

    # Train
    while epoch < config.max_epochs:
        model.train()
        bar = tqdm.tqdm(total=len(train_dl), smoothing=0)
        for idx, batch in enumerate(train_dl):
            metrics = model.update(
                    to_device(batch["obss"], config.device),
                    to_device(batch["masks"].permute(0,1,4,2,3), config.device),
                    step
            )
            wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)
            step += 1
            bar.update(1)

            if step % config.eval_interval == 0:
                model.eval()
                model, best_val_loss = eval_and_save(model, val_dl, epoch, step, best_val_loss, config)
                model.train()
        if hasattr(model, "scheduler"):
            self.scheduler.step()
        epoch += 1
        wandb.log({"epoch": epoch}, step=step)

    # wandb finish
    wandb.finish()

def eval_and_save(model, val_dl, epoch, step, best_val_loss, config):
    with torch.no_grad() if config.ocr.name != 'Iodine' else nullcontext():
        metrics = []
        for idx, batch in enumerate(val_dl):
            m = model.get_loss(
                    to_device(batch["obss"], config.device),
                    to_device(batch["masks"].permute(0,1,4,2,3), config.device),
            )
            # This is just for iodine since we can't use no_grad, but prevents us from using too much gpu
            m['loss'] = m['loss'].detach()
            metrics.append(m)
        metrics = {
            key: np.mean([get_item(metric[key]) for metric in metrics])
            for key in metrics[0]
        }
        best = False
        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"].item()
            best = True
        metrics.update({"best_loss": best_val_loss})
        wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=step)
        log.info(
            f"[Epoch {epoch}, Step {step}] "
            + " / ".join(f"val/{k} {v:.4f}" for k, v in metrics.items())
        )
    if best:
        samples = model.get_samples(
           to_device(batch["obss"], config.device)[: config.num_visualization]
        )
        wandb.log(
            {k: [wandb.Image(_v) for _v in v] for k, v in samples.items()},
            step=step,
        )
    if config.ocr.name == "SlotAttn":
        model.decay_lr(best)

    save(
        model,
        step=step,
        epoch=epoch,
        best_val_loss=best_val_loss,
        best=best,
    )

    return model, best_val_loss


if __name__ == "__main__":
    main()
