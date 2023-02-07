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

log = logging.getLogger(__name__)


@hydra.main(config_path="configs/", config_name="get_ari_mse")
def main(config):
    log_name = get_log_prefix(config)
    log_name += f"-{config.dataset.name}"
    init_wandb(config, "GetARI-" + log_name,
            tags=config.tags.split(",") + config.dataset.tags.split(",")
    )

    # Data Loader
    train_dl, val_dl = get_dataloaders(
        config.dataset,
        config.batch_size,
        config.num_workers,
    )

    # Model
    ocr = getattr(ocrs, config.ocr.name)(config.ocr, config.dataset)
    ocr.load(torch.load(get_ocr_checkpoint_path(config.ocr_checkpoint), map_location="cpu"))
    #ocr.wandb_watch(config.wandb)
    ocr.to(config.device)

    # calculate ARI
    ocr.eval()
    aris = []
    mses = []
    bar = tqdm.tqdm(total=len(val_dl), smoothing=0) # using validation data for getting ARI score
    with torch.no_grad():
        for idx, batch in enumerate(val_dl):
            obss = to_device(batch["obss"], config.device)
            gt_masks = to_device(batch["masks"].permute(0,1,4,2,3), config.device)

            # get mse and ari
            if config.ocr.name == "SLATE":
                metrics = ocr.get_loss(obss, masks=gt_masks, with_mse=True)
            else:
                metrics = ocr.get_loss(obss, masks=gt_masks)
            mses.append(get_item(metrics['mse']))
            aris.append(get_item(metrics['ari']))

            bar.update(1)

    ari = np.mean(aris)
    log.info(f"ARI is {ari}")
    wandb.log({"ARI": ari})

    mse = np.mean(mses)
    log.info(f"MSE is {mse}")
    wandb.log({"MSE": mse})

    # wandb finish
    wandb.finish()


if __name__ == "__main__":
    main()
