import tqdm
import torch
import wandb
import hydra
import logging
import omegaconf
from pathlib import Path
from torch.utils.data import DataLoader
import numpy as np

import ocrs
import poolings
from utils.tools import *
from utils.property_predictor import PropertyPredictor

log = logging.getLogger(__name__)


@hydra.main(config_path="configs/", config_name="train_property_predictor")
def main(config):
    log_name = get_log_prefix(config)
    log_name += f"-{config.dataset.name}"
    tags = config.tags.split(",") + config.dataset.tags.split(",")
    init_wandb(config, "TrainPropertyPredictor-" + log_name, tags=tags)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Data Loader
    train_dl, val_dl = get_dataloaders(
        config.dataset,
        config.batch_size,
        config.num_workers,
    )

    # Model
    ocr = getattr(ocrs, config.ocr.name)(config.ocr, config.dataset)
    ocr.load(torch.load(get_ocr_checkpoint_path(config.ocr_checkpoint), map_location="cpu"))
    model = PropertyPredictor(ocr, config.property_predictor, config.dataset)
    model.wandb_watch(config.wandb)
    model.to(config.device)

    # Train
    step, epoch, best_val_loss = 0, 0, 1e10
    while epoch < config.max_epochs:
        model.train()
        bar = tqdm.tqdm(total=len(train_dl), smoothing=0)
        for idx, batch in enumerate(train_dl):
            metrics = model.update(to_device(batch, config.device), step)
            wandb.log({f"train/{k}": v for k, v in metrics.items()}, step=step)
            step += 1
            bar.update(1)
        epoch += 1
        wandb.log({"epoch": epoch}, step=step)

        model.eval()
        with torch.no_grad():
            metrics = []
            for idx, batch in enumerate(val_dl):
                metrics.append(model.get_loss(to_device(batch, config.device)))
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
                f"[{epoch}] "
                + " / ".join(f"val/{k} {v:.4f}" for k, v in metrics.items())
            )
            if best:
                samples = model.get_samples(
                    to_device(batch, config.device)["obss"][: config.num_visualization]
                )
                wandb.log(
                    {k: [wandb.Image(_v) for _v in v] for k, v in samples.items()},
                    step=step,
                )

    # wandb finish
    wandb.finish()


if __name__ == "__main__":
    main()
