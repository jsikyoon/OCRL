import gym
import h5py
import tqdm
import wandb
import hydra
import argparse
import multiprocessing
import numpy as np
from PIL import Image

import envs
from utils.tools import *


def get_data(procidx, env, num_data, only_initial, return_dict):
    obss = []
    objs = []
    masks = []
    num_objs = []
    labels = []
    obs = env.reset()
    bar = tqdm.tqdm(total=num_data, smoothing=0)
    while len(obss) < num_data:
        if not only_initial:
            obs, _, done, info = env.step(env.action_space.sample())
        obss.append(obs)
        objs.append(env.render(mode="state", fill_empty=True))
        masks.append(env.render(mode="mask", fill_empty=True))
        num_objs.append(env._num_objects)
        labels.append(env._target_obj_idx)
        if not only_initial:
            if done:
                obs = env.reset()
        else:
            obs = env.reset()
        bar.update(1)
    return_dict[procidx] = {
        "obss": obss[:num_data],
        "objs": objs[:num_data],
        "masks": masks[:num_data],
        "num_objs": num_objs[:num_data],
        "labels": labels[:num_data],
    }


def parallel_get_data(env, num_data, num_proc, only_initial):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    processes = []
    for i in range(num_proc):
        p = multiprocessing.Process(
            target=get_data, args=(i, env[i], num_data // num_proc, only_initial, return_dict)
        )
        processes.append(p)
        p.start()
    for p in processes:
        p.join()
    obss, objs, masks, num_objs, labels = [], [], [], [], []
    for _, value in return_dict.items():
        obss.extend(value["obss"])
        objs.extend(value["objs"])
        masks.extend(value["masks"])
        num_objs.extend(value["num_objs"])
        labels.extend(value["labels"])
    return np.array(obss), np.array(objs), np.array(masks), np.array(num_objs), np.array(labels)


def create_datasets(f, group_name, num_samples, obss, objs, masks, num_objs, labels):
    group = f.create_group(group_name)
    group.create_dataset("obss", [num_samples]+list(obss.shape[1:]))
    group.create_dataset("objs", [num_samples]+list(objs.shape[1:]))
    group.create_dataset("masks", [num_samples]+list(masks.shape[1:]))
    group.create_dataset("num_objs", [num_samples]+list(num_objs.shape[1:]))
    group.create_dataset("labels", [num_samples]+list(labels.shape[1:]))


def generate_datasets(fname, group_name, num_samples, chunk_size, env, num_proc, only_initial):
    num_chunks = num_samples // chunk_size
    assert num_samples % chunk_size == 0
    for i in range(num_chunks):
        print(f"generating {group_name} {i*chunk_size}/{num_samples}")
        f = h5py.File(fname, "a")
        obss, objs, masks, num_objs, labels = parallel_get_data(env, chunk_size, num_proc, only_initial)

        #masks = np.transpose(np.concatenate([masks]*3, axis=-1), (0,2,1,3,4)) * 255
        #masks = np.reshape(masks, [masks.shape[0], masks.shape[1], -1, masks.shape[-1]])
        #vizs = np.concatenate([obss, masks], axis=2)
        #for i in range(vizs.shape[0]):
        #    wandb.log({'sample': [wandb.Image(vizs[i])]})
        #exit(1)

        assert len(obss) == chunk_size and len(labels) == chunk_size and len(objs) == chunk_size
        f[group_name]["obss"][i*chunk_size:(i+1)*chunk_size] = obss
        f[group_name]["labels"][i*chunk_size:(i+1)*chunk_size] = labels
        f[group_name]["objs"][i*chunk_size:(i+1)*chunk_size] = objs
        f[group_name]["masks"][i*chunk_size:(i+1)*chunk_size] = masks
        f[group_name]["num_objs"][i*chunk_size:(i+1)*chunk_size] = num_objs
        f.close()


@hydra.main(config_path="configs/", config_name="collect_dataset_from_synthetic_envs")
def main(config):
    num_tr = config.collection.num_tr
    num_val = config.collection.num_val
    num_proc = config.collection.num_proc
    only_initial = config.collection.only_initial
    #only_initial = True if config.collection.only_initial=='True' else False
    env = [getattr(envs, config.env.env)(config.env, seed=config.collection.seed+i)
            for i in range(num_proc)]
    if config.env.agent_pos is not None:
        agent_pos = str(config.env.agent_pos[0]).replace(".", "") + str(
            config.env.agent_pos[1]
        ).replace(".", "")
    else:
        agent_pos = "None"
    num_colors = len(env[0]._COLORS)
    num_shapes = len(env[0]._SHAPES)
    num_scales = len(env[0]._SCALES)
    num_tr = config.collection.num_tr
    num_val = config.collection.num_val
    chunk_size = config.collection.chunk_size
    log_name = (
        f"{config.env.env}-"
        f"N{config.env.num_objects_range[0]}-{config.env.num_objects_range[1]}C{num_colors}S{num_shapes}S{num_scales}-"
        f"{config.env.mode}Mode-"
        f"UseBG{config.env.background.use_bg}-"
        f"AgentPos{agent_pos}-WoAgent{config.env.wo_agent}-Occlusion{config.env.occlusion}-"
        f"Skewed{config.env.skewed}-Seed{config.collection.seed}-Tr{num_tr}-Val{num_val}"
    )
    init_wandb(
        config,
        "CollectData-" + log_name,
    )
    file_name = log_name + ".hdf5"
    dataset_dir = Path(wandb.run.dir)
    f = h5py.File(dataset_dir / file_name, "a")
    obss, objs, masks, num_objs, labels = parallel_get_data(env, num_proc, num_proc, only_initial)
    create_datasets(f, "TrainingSet", num_tr, obss, objs, masks, num_objs, labels)
    create_datasets(f, "ValidationSet", num_val, obss, objs, masks, num_objs, labels)
    f.close()
    if num_tr < chunk_size:
        generate_datasets(dataset_dir/file_name, "TrainingSet", num_tr, num_tr, env, num_proc, only_initial)
    else:
        generate_datasets(dataset_dir/file_name, "TrainingSet", num_tr, chunk_size, env, num_proc, only_initial)
    if num_val < chunk_size:
        generate_datasets(dataset_dir/file_name, "ValidationSet", num_val, num_val, env, num_proc, only_initial)
    else:
        generate_datasets(dataset_dir/file_name, "ValidationSet", num_val, chunk_size, env, num_proc, only_initial)

    # wandb finish
    wandb.finish()


if __name__ == "__main__":
    main()
