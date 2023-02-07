import argparse
import multiprocessing
import os

import cw_envs
import gym
import h5py
import hydra
import numpy as np
import tqdm
from PIL import Image

num_tr = 1000000
num_val = 10000
num_proc = 50

def get_data(procidx, env, num_data, return_dict):
    obss = []
    # objs = []
    num_objs = []
    labels = []
    # obs = env.reset()
    obs = env.reset()
    bar = tqdm.tqdm(total=num_data, smoothing=0)
    while len(obss) < num_data:
        obs, _, done, info = env.step(env.action_space.sample())
        num_ch = obs.shape[-1]
        # every 3 channels is an image
        for i in range(num_ch // 3):
            obss.append(obs[..., (i*3):(i*3)+3])
            num_objs.append(env.num_objects)
            labels.append(env.target_obj_idx)
        if done:
            obs = env.reset()
        bar.update(1)
    return_dict[procidx] = {
        "obss": obss[:num_data],
        # "objs": objs[:num_data],
        "num_objs": num_objs[:num_data],
        "labels": labels[:num_data],
    }


def parallel_get_data(env, num_data):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    i = 0
    for i in range(num_proc):
        p = multiprocessing.Process(
            target=get_data, args=(i, env[i], num_data // num_proc, return_dict)
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    obss, objs, num_objs, labels = [], [], [], []
    for _, value in return_dict.items():
        obss.extend(value["obss"])
        # objs.extend(value["objs"])
        num_objs.extend(value["num_objs"])
        labels.extend(value["labels"])
    return obss, objs, num_objs, labels


@hydra.main(config_path="../configs/env", config_name="cw-fingerimg-notarget-N4C11S1S1-hard")
def main(config):
    env = [getattr(cw_envs, config.env)(config, seed=i) for i in range(num_proc)]
    num_colors = len(config.COLORS)
    file_name = f"{config.env}-" f"N4C{num_colors}S1S1-Tr{num_tr}-Val{num_val}.hdf5"
    f = h5py.File(file_name, "w")
    tr_group = f.create_group("TrainingSet")
    obss, objs, num_objs, labels = parallel_get_data(env, num_tr)
    # assert len(obss) == num_tr and len(labels) == num_tr and len(objs) == num_tr
    tr_group["obss"] = obss
    # tr_group["objs"] = objs
    tr_group["num_objs"] = num_objs
    tr_group["labels"] = labels
    val_group = f.create_group("ValidationSet")
    obss, objs, num_objs, labels = parallel_get_data(env, num_val)
    # assert len(obss) == num_val and len(labels) == num_val and len(objs) == num_val
    val_group["obss"] = obss
    # val_group["objs"] = objs
    val_group["num_objs"] = num_objs
    val_group["labels"] = labels
    print("done", os.getcwd(), file_name)
    f.close()


if __name__ == "__main__":
    main()
