# OCRL
Object-Centric-Representation Library (OCRL): This repository is to explore OCR on various downstream tasks from supervised learning tasks to RL tasks

## Quick Start
Install required packages
```
pip install -r requirements.txt
```

Because we support multiple options, the number of running hyperparameters is huge. To run multiple variations, the details are written in `for_running.json`, and we will give examples to run the representation module pre-training and agent learning.

To pre-train representation modules, after downloading [the dataset](https://www.dropbox.com/sh/hr3hg73ybrraftm/AAAog0bSEJwPOz75_gkbGzbfa?dl=0), you can do like
```
python train_ocr.py ocr=slate ocr.slotattr.num_slots=6 ocr.slotattr.num_iterations=6 dataset=random-N5C4S4S2 device=cuda:0
python train_ocr.py ocr=vae ocr.cnn_feat_size=4 ocr.use_cnn_feat=False dataset=random-N5C4S4S2 device=cuda:1 ocr.learning.kld_weight=5 device=cuda:0
```

To learn agent-learning, you can learn the models by running `python run_sb3s.py` with your configurations written in `for_running.json` such as
```
...
models = [
        "e2e cnn 0.01ent_coef",
        "slate-transformer-0.01ent_coef",
]
...
envs = [
        "targetN4-hard-sparse",
        "pushN3-hard-sparse",
        "oooC2S2S1-hard-sparse-oc",
        "oooC2S2S1-hard-sparse",
]
...
```
Pretrained SLATE [5] is included in this repository, and other pretrained encoders can be downloaded through [this link](https://www.dropbox.com/sh/hr3hg73ybrraftm/AAAog0bSEJwPOz75_gkbGzbfa?dl=0).

## Representation Modules
We implemented not just OCR algorithms, but also single vector representations, CNN feature map, MAE patch representations, and ground truth wrapper are supported to compare with OCRs.

Implemented representation modules
- Ground Truth states
- VAE [1]
- CNN [2]
- MAE [3]
- Slot-Attention [4]
- SLATE [5]
- IODINE [6]

## Pooling layers
To utilize diverse representations, we implement a variety of pooling modules. On pooling layer, we can control to train OCR through downstream task loss or OCR loss or both of them.

- Transformer: Using additional CLS token to represent the slots [7]
- Relation Network [8]
- MLP
- CNN-Linear: Using CNN architecture and one linear layer [9]
- CNN-Transformer: Using CNN architecture and encode the patch representations through Transformer
- Identity (Concatenate): If the representation is a single vector, it doesn't do anything, if the representation consists of multiple vectors, it concatenates them.

## Downstream models
This codes support to use OCR for supervised learning and agent learning.

- MLP classifier
- Property Prediction
- ARI, MSE calculation
- RL Agents (It is based on Stable-baselines3 [10])
    - PPO

## Contact
Any feedback are welcome! Please open an issue on this repository or send email to Jaesik Yoon (mail@jaesikyoon.com).

## Reference

[1] Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).

[2] Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." nature 518.7540 (2015): 529-533.

[3] He, Kaiming, et al. "Masked autoencoders are scalable vision learners." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

[4] Locatello, Francesco, et al. "Object-centric learning with slot attention." Advances in Neural Information Processing Systems 33 (2020): 11525-11538.

[5] Singh, Gautam, Fei Deng, and Sungjin Ahn. "Illiterate dall-e learns to compose." arXiv preprint arXiv:2110.11405 (2021).

[6] Greff, Klaus, et al. "Multi-object representation learning with iterative variational inference." International Conference on Machine Learning. PMLR, 2019.

[7] Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).

[8] Santoro, Adam, et al. "A simple neural network module for relational reasoning." Advances in neural information processing systems 30 (2017).

[9] Heravi, Negin, et al. "Visuomotor control in multi-object scenes using object-aware representations." arXiv preprint arXiv:2205.06333 (2022).

[10] Raffin, Antonin, et al. "Stable baselines3." (2019).
