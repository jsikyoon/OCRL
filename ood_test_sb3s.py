import os
import datetime

session_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
available_gpus = [0,1,2,3,4,5,6,7]
agent_entity = "ocrl_benchmark"
agent_project = "agent-learning"
res_entity = "jaesikyoon"
res_project = "ocrl-ood"
task = "push-N3C4S1S1"


if task == "target-N4C4S3S1":  # object-goal task
    ids = {
        "gt": ["3cuaev2v", "2g6s2u4i", "2lu2841p"],
        "cnn": ["722afroa", "1kpsy7i7", "2h73jlmx"],
        "cnnfeat": ["lkpyhbai", "350o8mhu", "in3g045u"],
        "multicnns": ["2n2ee0wh", "34027uo0", "rqolsnra"],
        "vae": ["3bui9wsq", "3nauvg8d", "n3bpsnaa"],
        "mae-cls": ["18n9fgx0", "j228ggkp", "qhgma1u1"],
        "slate-cnn": ["1jy1921s", "1jwar3k9", "3glevf8r"],
        "mae-patch": ["3ka72x54", "34tv9lag", "uy5z7bpy"],
        "slate": ["2jtqf9nl", "381f2oav", "1i5l2rsc"],
        "sa": ["3p408ujb", "1pc4fcjk", "24n0rdg7"],
        "sa-large": ["dspdke44", "1abhga4e", "18kgm2ws"],
        "iodine": ["3d9wv72x", "1mhl9btj", "xgcom9q8"],
    }
    envs = [
        "target-N3C4S3S1",
        "target-N4C4S3S1",
        "target-N5C4S3S1",
        "target-N6C4S3S1",
        "target-N4C4S3S1-ood-ocr-color1",
        "target-N4C4S3S1-ood-ocr-color2",
        "target-N4C4S3S1-ood-ocr-color3",
    ]
elif task == "push-N3C4S1S1":  # object-interaction task
    ids = {
        "gt": ["3r2xbhoa", "3238yw46", "gu3vhf4f"],
        "cnn": ["nyfnpj42", "3na7t6jn", "38nhjdzb"],
        "cnnfeat": ["36si14y2", "3nlnase6", "ce7bh36n"],
        "multicnns": ["2f5hwglq", "tzkuentr", "103vg809"],
        "vae": ["1n06686m", "mbgl8hg1", "1zvqjxyg"],
        "mae-cls": ["knkmgntd", "e28l9ag8", "2dbu6zyt"],
        "slate-cnn": ["1xbi21b2", "13zmrsd8", "2gnj0sk8"],
        "mae-patch": ["3iwiz0dj", "35plejy9", "3fvym3i0"],
        "slate": ["17j7uj00", "20mgxoio", "2jdndm72"],
        "sa": ["3v4hfph9", "3uv1ss0r", "1eap4o97"],
        "sa-large": ["2iedbvjo", "vb9001qo", "b8omgo8u"],
        "iodine": ["2zt784fd", "2rocmh4y", "342i394z"],
    }
    envs = [
        "push-N1C4S1S1",
        "push-N2C4S1S1",
        "push-N3C4S1S1",
        "push-N4C4S1S1",
        "push-N3C4S1S1-ood-ocr-color1",
        "push-N3C4S1S1-ood-ocr-color2",
        "push-N3C4S1S1-ood-ocr-color3",
    ]
elif task == "ood-one-out-N4C2S2S1-oc":  # object-comparison task
    ids = {
         "gt": ["3az9t1q6", "241wk6k7", "k2yj4fus"],
         "cnn": ["tte4qpmg", "3m0k9tgq", "2ckrv1tr"],
         "cnnfeat": ["2po8gs0r", "1xzo35rm", "h14d102j"],
         "multicnns": ["1m2my6mp", "k3ivkuqj", "2najwc2q"],
         "vae": ["3iev7zwz", "pkir2zoi", "1jn4j1pr"],
         "mae-cls": ["10swmi78", "3gwhbkl4", "1l2dawbq"],
         "slate-cnn": ["2ag9gwxg", "2m1vva30", "1mhke2b9"],
         "mae-patch": ["dva9hrht", "3ojkcnyi", "34x4ltas"],
         "slate": ["qg5gsaxi", "3h5nw8r8", "3lkk3c1h"],
         "sa": ["9f2ifwb0", "358v8ctb", "1inbulsi"],
         "sa-large": ["1f94knww", "16c574un", "3khzutm5"],
         "iodine": ["4zeessti", "1934f3ly", "3h6w4awq"],
    }
    envs = [
         "odd-one-out-N3C2S2S1-oc",
         "odd-one-out-N4C2S2S1-oc",
         "odd-one-out-N5C2S2S1-oc",
         "odd-one-out-N6C2S2S1-oc",
         "odd-one-out-N4C2S2S1-oc-ood-agent-shape1",
         "odd-one-out-N4C2S2S1-oc-ood-agent-shape2",
         "odd-one-out-N4C2S2S1-oc-ood-agent-color1",
         "odd-one-out-N4C2S2S1-oc-ood-agent-color2",
         "odd-one-out-N4C2S2S1-oc-ood-ocr-shape1",
         "odd-one-out-N4C2S2S1-oc-ood-ocr-shape2",
         "odd-one-out-N4C2S2S1-oc-ood-ocr-color1",
         "odd-one-out-N4C2S2S1-oc-ood-ocr-color2",
    ]
elif task == "ood-one-out-N4C2S2S1":  # property-comparison task
    ids = {
         "gt": ["o9cw8zrp", "arkliovu", "16dgbu3y"],
         "cnn": ["2yxhgz4b", "nyc6qru3", "b4fje1sv"],
         "cnnfeat": ["2hztrz0w", "1krm6ycy", "2deu6ben"],
         "multicnns": ["3kzzmr3w", "1sccyrzz", "2do7hb2w"],
         "vae": ["1v8wgpoe", "32b2xuor", "4h3a7te5"],
         "mae-cls": ["30xmm2ri", "ywuwejtj", "2f7gto6b"],
         "slate-cnn": ["q8ylnb23", "1l1yu9w0", "1i2nwd7k"],
         "mae-patch": ["3fynupkd", "2d4p5k60", "1fiu6qwz"],
         "slate": ["270y8iki", "11heyik8", "5fhwbfv9"],
         "sa": ["2olswe0l", "2nijqs8b", "2fk5xhvm"],
         "sa-large": ["2vh2gxwv", "2fqgg1d6", "1yrkg4vv"],
         "iodine": ["nk7dhg85", "4767kx25", "1rh6iy0y"],
    }
    envs = [
         "odd-one-out-N3C2S2S1",
         "odd-one-out-N4C2S2S1",
         "odd-one-out-N5C2S2S1",
         "odd-one-out-N6C2S2S1",
         "odd-one-out-N4C2S2S1-ood-agent-shape1",
         "odd-one-out-N4C2S2S1-ood-agent-shape2",
         "odd-one-out-N4C2S2S1-ood-agent-color1",
         "odd-one-out-N4C2S2S1-ood-agent-color2",
         "odd-one-out-N4C2S2S1-ood-ocr-shape1",
         "odd-one-out-N4C2S2S1-ood-ocr-shape2",
         "odd-one-out-N4C2S2S1-ood-ocr-color1",
         "odd-one-out-N4C2S2S1-ood-ocr-color2",
    ]
else:
    raise ValueError(f"Unknown task {task}")


# create tmux session
os.system(f"tmux new-session -s {session_name} -d")
cnt = 0
for env in envs:
    for model, model_ids in ids.items():
        for model_id in model_ids:
            dev = available_gpus[cnt % len(available_gpus)]
            additional_args = f"device={dev}"
            os.system(f"tmux new-window -t {session_name}")
            if model == "slate":
                num_slots = int(env.split("N")[1][0]) + 2
                if env == "target-N4C4S3S1":
                    command = ( # for target task
                        f"python test_sb3.py ocr=slate ocr.slotattr.num_slots={num_slots} "
                        f"pooling=transformer sb3=ppo sb3_acnet=mlp env={env} "
                        f"pooling.num_layers=3 "
                        f"n_eval_episodes=100 "
                        f"pooling.ocr_checkpoint.entity=ocrl_benchmark "
                        f"pooling.ocr_checkpoint.project=pre-training "
                        f"pooling.ocr_checkpoint.run_id=4rrrm4gx "
                        f"pooling.ocr_checkpoint.file=checkpoints/model_333336.pth "
                        f"agent_checkpoint.entity={agent_entity} "
                        f"agent_checkpoint.project={agent_project} "
                        f"agent_checkpoint.run_id={model_id} "
                        f"tags=\\'slate,{env}\\' "
                    )
                else:
                    command = (
                        f"python test_sb3.py ocr=slate ocr.slotattr.num_slots={num_slots} "
                        f"pooling=transformer sb3=ppo sb3_acnet=mlp env={env} "
                        f"pooling.num_layers=1 "
                        f"n_eval_episodes=100 "
                        f"pooling.ocr_checkpoint.entity=ocrl_benchmark "
                        f"pooling.ocr_checkpoint.project=pre-training "
                        f"pooling.ocr_checkpoint.run_id=4rrrm4gx "
                        f"pooling.ocr_checkpoint.file=checkpoints/model_333336.pth "
                        f"agent_checkpoint.entity={agent_entity} "
                        f"agent_checkpoint.project={agent_project} "
                        f"agent_checkpoint.run_id={model_id} "
                        f"tags=\\'slate,{env}\\' "
                    )
            elif model == "sa":
                num_slots = int(env.split("N")[1][0]) + 2
                command = (
                    f"python test_sb3.py ocr=slate ocr.slotattr.num_slots={num_slots} "
                    f"ocr.use_bcdec=True ocr.slotattr.slot_size=64 ocr.slotattr.mlp_hidden_size=128 "
                    f"ocr.slotattr.num_iterations=7 "
                    f"pooling=transformer sb3=ppo sb3_acnet=mlp env={env} "
                    f"pooling.num_layers=1 "
                    f"n_eval_episodes=100 "
                    f"pooling.ocr_checkpoint.entity=ocrl_benchmark "
                    f"pooling.ocr_checkpoint.project=pre-training "
                    f"pooling.ocr_checkpoint.run_id=29dh0co7 "
                    f"pooling.ocr_checkpoint.file=checkpoints/model_best.pth "
                    f"agent_checkpoint.entity={agent_entity} "
                    f"agent_checkpoint.project={agent_project} "
                    f"agent_checkpoint.run_id={model_id} "
                    f"tags=\\'slotattention,{env}\\' "
                )
            elif model == "sa-large":
                num_slots = int(env.split("N")[1][0]) + 2
                if env == "ood-one-out-N4C2S2S1oc":
                    command = ( # for object-comparison task
                        f"python test_sb3.py ocr=slate ocr.slotattr.num_slots={num_slots} "
                        f"ocr.use_bcdec=True "
                        f"ocr.slotattr.num_iterations=7 "
                        f"pooling=transformer sb3=ppo sb3_acnet=mlp env={env} "
                        f"pooling.num_layers=1 "
                        f"n_eval_episodes=100 "
                        f"pooling.ocr_checkpoint.entity=ocrl_benchmark "
                        f"pooling.ocr_checkpoint.project=pre-training "
                        f"pooling.ocr_checkpoint.run_id=1jaqzsd8 "
                        f"pooling.ocr_checkpoint.file=checkpoints/model_best.pth "
                        f"agent_checkpoint.entity={agent_entity} "
                        f"agent_checkpoint.project={agent_project} "
                        f"agent_checkpoint.run_id={model_id} "
                        f"tags=\\'slotattention-large,{env}\\' "
                    )
                else:
                    command = (
                        f"python test_sb3.py ocr=slate ocr.slotattr.num_slots={num_slots} "
                        f"ocr.use_bcdec=True "
                        f"ocr.slotattr.num_iterations=3 "
                        f"pooling=transformer sb3=ppo sb3_acnet=mlp env={env} "
                        f"pooling.num_layers=1 "
                        f"n_eval_episodes=100 "
                        f"pooling.ocr_checkpoint.entity=ocrl_benchmark "
                        f"pooling.ocr_checkpoint.project=pre-training "
                        f"pooling.ocr_checkpoint.run_id=1bmfg6ro "
                        f"pooling.ocr_checkpoint.file=checkpoints/model_best.pth "
                        f"agent_checkpoint.entity={agent_entity} "
                        f"agent_checkpoint.project={agent_project} "
                        f"agent_checkpoint.run_id={model_id} "
                        f"tags=\\'slotattention-large,{env}\\' "
                    )
            elif model == "iodine":
                num_slots = int(env.split("N")[1][0]) + 2
                command = (
                    f"python test_sb3.py ocr=iodine_large ocr.num_slots={num_slots} "
                    f"pooling=transformer sb3=ppo sb3_acnet=mlp env={env} "
                    f"pooling.num_layers=1 "
                    f"n_eval_episodes=100 "
                    f"pooling.ocr_checkpoint.entity=ocrl_benchmark "
                    f"pooling.ocr_checkpoint.project=pre-training "
                    f"pooling.ocr_checkpoint.run_id=83o0niqd "
                    f"pooling.ocr_checkpoint.file=checkpoints/model_best.pth "
                    f"agent_checkpoint.entity={agent_entity} "
                    f"agent_checkpoint.project={agent_project} "
                    f"agent_checkpoint.run_id={model_id} "
                    f"tags=\\'iodine,{env}\\' "
                )
            elif model == "cnn":
                command = (
                    f"python test_sb3.py ocr=naturecnn pooling=identity sb3=ppo "
                    f"sb3_acnet=identity_orthoinit env={env} "
                    f"n_eval_episodes=100 "
                    f"agent_checkpoint.entity={agent_entity} "
                    f"agent_checkpoint.project={agent_project} "
                    f"agent_checkpoint.run_id={model_id} "
                    f"tags=\\'cnn,{env}\\' "
                )
            elif model == "cnnfeat":
                command = (
                    f"python test_sb3.py ocr=naturecnn ocr.use_cnn_feat=True "
                    f"pooling=transformer pooling.pos_emb=ape sb3=ppo "
                    f"sb3_acnet=mlp env={env} "
                    f"n_eval_episodes=100 "
                    f"agent_checkpoint.entity={agent_entity} "
                    f"agent_checkpoint.project={agent_project} "
                    f"agent_checkpoint.run_id={model_id} "
                    f"tags=\\'cnnfeat,{env}\\' "
                )
            elif model == "multicnns":
                command = (
                    f"python test_sb3.py ocr=multiple_cnn "
                    f"pooling=transformer pooling.pos_emb=ape sb3=ppo "
                    f"sb3_acnet=mlp env={env} "
                    f"n_eval_episodes=100 "
                    f"agent_checkpoint.entity={agent_entity} "
                    f"agent_checkpoint.project={agent_project} "
                    f"agent_checkpoint.run_id={model_id} "
                    f"tags=\\'multicnns,{env}\\' "
                )
            elif model == "vae":
                command = (
                    f"python test_sb3.py ocr=vae pooling=mlp sb3=ppo "
                    f"sb3_acnet=mlp env={env} "
                    f"n_eval_episodes=100 "
                    f"pooling.ocr_checkpoint.entity=ocrl_benchmark "
                    f"pooling.ocr_checkpoint.project=pre-training "
                    f"pooling.ocr_checkpoint.run_id=5p35rn99 "
                    f"pooling.ocr_checkpoint.file=checkpoints/model_best.pth "
                    f"agent_checkpoint.entity={agent_entity} "
                    f"agent_checkpoint.project={agent_project} "
                    f"agent_checkpoint.run_id={model_id} "
                    f"tags=\\'vae,{env}\\' "
                )
            elif model == "mae-cls":
                command = (
                    f"python test_sb3.py ocr=mae ocr.patch_size=16 ocr.masking_ratio=0.0 ocr.return_cls=True "
                    f"pooling=mlp sb3=ppo "
                    f"sb3_acnet=mlp env={env} "
                    f"n_eval_episodes=100 "
                    f"pooling.ocr_checkpoint.local_file=trained_models/mae-vitb-patch16/19ve6sfu/model_best.pth "
                    f"agent_checkpoint.entity={agent_entity} "
                    f"agent_checkpoint.project={agent_project} "
                    f"agent_checkpoint.run_id={model_id} "
                    f"tags=\\'mae-cls,{env}\\' "
                )
            elif model == "slate-cnn":
                command = (
                    f"python test_sb3.py ocr=slate ocr.use_cnn_feat=True "
                    f"pooling=cnn_linear sb3=ppo sb3_acnet=mlp env={env} "
                    f"n_eval_episodes=100 "
                    f"pooling.ocr_checkpoint.entity=ocrl_benchmark "
                    f"pooling.ocr_checkpoint.project=pre-training "
                    f"pooling.ocr_checkpoint.run_id=4rrrm4gx "
                    f"pooling.ocr_checkpoint.file=checkpoints/model_333336.pth "
                    f"agent_checkpoint.entity={agent_entity} "
                    f"agent_checkpoint.project={agent_project} "
                    f"agent_checkpoint.run_id={model_id} "
                    f"tags=\\'slate-cnn,{env}\\' "
                )
            elif model == "mae-patch":
                command = (
                    f"python test_sb3.py ocr=mae ocr.patch_size=16 ocr.masking_ratio=0.0 "
                    f"pooling=transformer pooling.num_layers=3 pooling.pos_emb=ape sb3=ppo "
                    f"sb3_acnet=mlp env={env} "
                    f"n_eval_episodes=100 "
                    f"pooling.ocr_checkpoint.local_file=trained_models/mae-vitb-patch16/19ve6sfu/model_best.pth "
                    f"agent_checkpoint.entity={agent_entity} "
                    f"agent_checkpoint.project={agent_project} "
                    f"agent_checkpoint.run_id={model_id} "
                    f"tags=\\'mae-patch,{env}\\' "
                )
            elif model == "gt":
                if env == "target-N4C4S3S1":
                    command = (  # for push task
                        f"python test_sb3.py ocr=gt sb3=ppo "
                        f"ocr.dims=\\[32,32\\] "
                        f"ocr.acts=\\['relu','relu'\\] "
                        f"pooling=mlp "
                        f"sb3_acnet=mlp env={env} "
                        f"n_eval_episodes=100 "
                        f"agent_checkpoint.entity={agent_entity} "
                        f"agent_checkpoint.project={agent_project} "
                        f"agent_checkpoint.run_id={model_id} "
                        f"tags=\\'gt,{env}\\' "
                    )
                else:
                    command = (
                        f"python test_sb3.py ocr=gt sb3=ppo "
                        f"pooling=transformer "
                        f"pooling.num_layers=3 "
                        f"sb3_acnet=mlp env={env} "
                        f"n_eval_episodes=100 "
                        f"agent_checkpoint.entity={agent_entity} "
                        f"agent_checkpoint.project={agent_project} "
                        f"agent_checkpoint.run_id={model_id} "
                        f"tags=\\'gt,{env}\\' "
                    )
            else:
                raise ValueError(f"{model} is not implemented yet")
            
            print(f"===================== {model} for {env} ======================")
            os.system(
                f"""tmux send-keys -t {session_name}:{cnt+1} "{command} {additional_args} wandb.entity={res_entity} wandb.project={res_project} " Enter"""
            )
            cnt += 1
