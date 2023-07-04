import os
import time
import json
import datetime

session_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

available_gpus = [1,2,3,4,5,6,7]
seeds = [1234,2345,3456]

with open("for_running.json", "r") as f:
    confs = json.load(f)

models = [
        "e2e cnn 0.01ent_coef",
        "slate-transformer-0.01ent_coef",
]
for m_name in models:
    if not m_name in confs["ocrs"].keys():
        raise ValueError(f"model {m_name} is not predefined. Please use in {confs['ocrs'].keys()}.")

envs = [
        "targetN4-hard-sparse", # Object Goal Task
        "pushN3-hard-sparse", # Object Interaction Task
        "oooC2S2S1-hard-sparse-oc", # Object Comparison Task
        "oooC2S2S1-hard-sparse", # Property Comparison Task
]
for e_name in envs:
    if not e_name in confs["envs"].keys():
        raise ValueError(f"env {e_name} is not predefined. Please use in {confs['envs'].keys()}.")


# create tmux session
os.system(f"tmux kill-session -t {session_name}")
os.system(f"tmux new-session -s {session_name} -d")

win_idx = 0
cnt = 0
for m_name in models:
    model_conf = confs["ocrs"][m_name]
    command = "python train_sb3.py "
    for key, value in model_conf.items():
        command += f"{key}={value} "
    for e_idx, e_name in enumerate(envs):
        os.system(f"tmux new-window -t {session_name}")
        for s_idx, _seed in enumerate(seeds):
            dev = available_gpus[cnt % len(available_gpus)]
            additional_args = f"device={dev} "
            env_conf = confs["envs"][e_name]
            for key, value in env_conf.items():
                additional_args += f"{key}={value} "
            os.system(f"tmux split-window -v -p 140 -t {session_name}:{win_idx+1}")
            print(f"{command} {additional_args} seed={_seed}")
            os.system( f"""tmux send-keys -t {session_name}:{win_idx+1}.{s_idx+1} "{command} {additional_args} seed={_seed}" Enter"""
            )
            cnt += 1
            time.sleep(10)
        os.system(f'tmux send-keys -t {session_name}:{win_idx+1}.0 "exit" Enter')
        os.system(f"tmux select-layout -t {session_name}:{win_idx+1} even-horizontal")
        win_idx += 1
os.system(f'tmux send-keys -t {session_name}:0 "exit" Enter')
