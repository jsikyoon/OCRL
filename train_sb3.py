import logging
from pathlib import Path

import hydra
import omegaconf
import stable_baselines3 as sb3
import wandb
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (DummyVecEnv, SubprocVecEnv,
                                              VecVideoRecorder)
from wandb.integration.sb3 import WandbCallback

import envs
import sb3s
from utils.tools import *

log = logging.getLogger(__name__)


@hydra.main(config_path="configs/", config_name="train_sb3")
def main(config):
    log_name = get_log_prefix(config)
    log_name += (
        f"-{config.sb3.name}-{config.sb3_acnet.name}-"
        f"{config.env.name}{config.env.mode}mode{config.env.rew_type}rewardtype-"
        f"Seed{config.seed}"
    )
    tags = config.tags.split(",") + config.env.tags.split(",") + [f"RandomSeed{config.seed}"]
    init_wandb(
        config,
        "TrainSB3-" + log_name,
        tags=tags,
        sync_tensorboard=True,
        monitor_gym=True,
    )

    if config.num_envs == 1:
        def make_env(seed=0):
            if config.ocr.name == "GT":
                config.env.render_mode = "state"
            env = getattr(envs, config.env.env)(config.env, seed)
            env = Monitor(env)  # record stats such as returns
            return env
        env = DummyVecEnv([make_env])
    else:
        def make_env(rank, seed=0):
            """
            Utility function for multiprocessed env.
                :param seed: (int) the inital seed for RNG
                :param rank: (int) index of the subprocess
            """
            def _init():
                if config.ocr.name == "GT":
                    config.env.render_mode = "state"
                env = getattr(envs, config.env.env)(config.env, rank + seed)
                env = Monitor(env)  # record stats such as returns
                return env
            set_random_seed(seed)
            return _init
        env = SubprocVecEnv(
            [make_env(i, seed=config.seed) for i in range(config.num_envs)],
            start_method="fork",
        )
    env = VecVideoRecorder(
        env,
        f"{wandb.run.dir}/videos/",
        record_video_trigger=lambda x: x % config.video.interval == 0,
        video_length=config.video.length,
    )
    if config.ocr.name == "GT":
        config.env.render_mode = "state"
    eval_env = getattr(envs, config.env.env)(
        config.env, seed=config.seed + config.num_envs
    )
    eval_env = Monitor(eval_env)  # record stats such as returns
    model_kwargs = {
        "verbose": 1,
        "tensorboard_log": f"{wandb.run.dir}/tb_logs/",
        "device": config.device,
        "policy_kwargs": dict(
            features_extractor_class=sb3s.OCRExtractor,
            features_extractor_kwargs=dict(config=config),
        ),
    }
    if hasattr(config.sb3, 'algo_kwargs'):
        model_kwargs = dict(model_kwargs, **config.sb3.algo_kwargs)
    if 'n_steps' in model_kwargs:
        model_kwargs['n_steps'] = model_kwargs['n_steps'] // config.num_envs
    policy = sb3s.CustomActorCriticPolicy
    model_kwargs['policy_kwargs']['config'] = config
    if hasattr(config.sb3, 'algo_kwargs'):
        model_kwargs = dict(model_kwargs, **config.sb3.algo_kwargs)
    if 'n_steps' in model_kwargs:
        model_kwargs['n_steps'] = model_kwargs['n_steps'] // config.num_envs
    model = getattr(sb3, config.sb3.name)(
        policy,
        env,
        **model_kwargs,
    )
    model.learn(
        total_timesteps=config.max_steps,
        callback=[
            WandbCallback(
                gradient_save_freq=config.wandb.log_gradient_freq,
                verbose=2,
            ),
            EvalCallback(
                eval_env,
                eval_freq=config.eval.freq,
                n_eval_episodes=config.eval.n_episodes,
                best_model_save_path=f"{wandb.run.dir}/models/",
                log_path=f"{wandb.run.dir}/eval_logs/",
                deterministic=False,
            ),
        ],
    )
    # wandb finish
    wandb.finish()


if __name__ == "__main__":
    main()
