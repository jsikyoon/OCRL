import os
import json
import time
import datetime

session_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")

available_gpus = [0,1,2,3,4,5,6,7]

with open("for_running.json", "r") as f:
    confs = json.load(f)

models = [
        "slate",
        "sa",
        "sa-large-slotiter3",
        "iodine",
]
for m_name in models:
    if not m_name in confs["ocrs_for_ari_mse_pp"].keys():
        raise ValueError(f"model {m_name} is not predefined. Please use in {confs['ocrs'].keys()}.")

dss = [
        "target",
        "oooC2S2oc",
        "oooC2S2",
        "push"
]
for d_name in dss:
    if not d_name in confs["datasets"].keys():
        raise ValueError(f"dataset {d_name} is not predefined. Please use in {confs['datasets'].keys()}.")

# create tmux session
os.system(f"tmux kill-session -t {session_name}")
os.system(f"tmux new-session -s {session_name} -d")

win_idx = 0
cnt = 0
for m_name in models:
    model_conf = confs["ocrs_for_pp"][m_name]
    command = "python train_property_predictor.py "
    for key, value in model_conf.items():
        command += f"{key}={value} "
    os.system(f"tmux new-window -t {session_name}")
    for d_idx, d_name in enumerate(dss):
        dev = available_gpus[cnt % len(available_gpus)]
        additional_args = f"device={dev} "
        ds_conf = confs["datasets"][d_name]
        for key, value in ds_conf.items():
            additional_args += f"{key}={value} "
        os.system(f"tmux split-window -v -p 140 -t {session_name}:{win_idx+1}")
        print(f"{command} {additional_args}")
        os.system( f"""tmux send-keys -t {session_name}:{win_idx+1}.{d_idx+1} "{command} {additional_args}" Enter"""
        )
        cnt += 1
        time.sleep(10)
    os.system(f'tmux send-keys -t {session_name}:{win_idx+1}.0 "exit" Enter')
    os.system(f"tmux select-layout -t {session_name}:{win_idx+1} even-horizontal")
    win_idx += 1
os.system(f'tmux send-keys -t {session_name}:0 "exit" Enter')
