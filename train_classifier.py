from contextlib import nullcontext
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
from utils.classifier import Classifier

log = logging.getLogger(__name__)


@hydra.main(config_path="configs/", config_name="train_classifier")
def main(config):
    log_name = get_log_prefix(config)
    log_name += f"-{config.dataset.name}"
    tags = config.tags.split(",") + config.dataset.tags.split(",")
    init_wandb(config, "TrainClassifier-" + log_name, tags=tags)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Data Loader
    train_dl, val_dl = get_dataloaders(
        config.dataset,
        config.batch_size,
        config.num_workers,
    )

    # Model
    ocr, _ = get_ocr(config.ocr, config.dataset, config.pooling.ocr_checkpoint, config.device)
    pooling = getattr(poolings, config.pooling.name + "_Module")(ocr.rep_dim, ocr.num_slots, config.pooling)
    model = Classifier(ocr, pooling, config.ocr.name, config.classifier, config.dataset.num_labels)
    model.to(config.device)

    # load
    step, epoch, best_val_loss = load(model)

    # Train
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
        with torch.no_grad() if config.ocr.name != 'Iodine' else nullcontext():
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
            if best and config.ocr.name != "GT" and config.ocr.name != "NatureCNN" and config.ocr.name != "ViT":
                samples = model.get_samples(
                    to_device(batch, config.device)["obss"][: config.num_visualization])
                wandb.log(
                    {k: [wandb.Image(_v) for _v in v] for k, v in samples.items()},
                    step=step,
                )
        save(
            model,
            step=step,
            epoch=epoch,
            best_val_loss=best_val_loss,
            best=best,
        )

    # wandb finish
    wandb.finish()


if __name__ == "__main__":
    main()
