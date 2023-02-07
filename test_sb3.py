import wandb
import hydra
import logging
import numpy as np
from pathlib import Path
import stable_baselines3 as sb3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

import envs
from utils.tools import *

from sb3s import OCRExtractor

log = logging.getLogger(__name__)


@hydra.main(config_path="configs/", config_name="test_sb3")
def main(config):
    log_name = f"TestSB3-{config.agent_checkpoint.run_id}-{config.env.name}"
    init_wandb(config, log_name, sync_tensorboard=True, monitor_gym=True, tags=config.tags.split(","))

    if config.ocr.name == "GT":
        config.env.render_mode = "state"

    def make_env(seed=0):
        if config.ocr.name == "GT":
            config.env.render_mode = "state"
        env = getattr(envs, config.env.env)(config.env, seed)
        env = Monitor(env)  # record stats such as returns
        return env
    env = DummyVecEnv([make_env])
    env = VecVideoRecorder(
        env,
        f"{wandb.run.dir}/videos/",
        record_video_trigger=lambda x: x % config.video.interval == 0,
        video_length=config.video.length,
    )

    # Download agent checkpoint
    api = wandb.Api()
    run = api.run(
        f"{config.agent_checkpoint.entity}/"
        f"{config.agent_checkpoint.project}/"
        f"{config.agent_checkpoint.run_id}"
    )
    agent_dir = Path(wandb.run.dir)
    run.file(config.agent_checkpoint.file).download(root=agent_dir, replace=True)
    if config.ocr.name == "GT":
        model = getattr(sb3, config.sb3.name).load(
            agent_dir / config.agent_checkpoint.file.split(".")[0],
            custom_objects={
                "observation_space": env.observation_space,
                "policy_kwargs": {
                    "features_extractor_class": OCRExtractor,
                    "features_extractor_kwargs": dict(config=config),
                    "config": config,
                },
            },
            device=config.device,
        )
    else:
        model = getattr(sb3, config.sb3.name).load(
            agent_dir / config.agent_checkpoint.file.split(".")[0],
            custom_objects={
                "policy_kwargs": {
                    "features_extractor_class": OCRExtractor,
                    "features_extractor_kwargs": dict(config=config),
                    "config": config,
                }
            },
            device=config.device,
        )
    is_success_buffer = []

    def log_success_callback(locals_: dict, globals_: dict) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.
        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                is_success_buffer.append(maybe_is_success)

    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=config.n_eval_episodes,
        deterministic=False,
        callback=log_success_callback,
    )
    success_rate = np.mean(is_success_buffer)
    print(
        f"mean_reward={mean_reward:.2f} +/- {std_reward}, success_rate={100 * success_rate:.2f}%"
    )
    wandb.log(
        {
            "mean_reward": mean_reward,
            "std_reward": std_reward,
            "success_rate": success_rate,
        }
    )

    # wandb finish
    wandb.finish()


if __name__ == "__main__":
    main()
