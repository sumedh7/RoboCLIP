# Pytorch + StableBaselines3 Implementation of RoboCLIP
This repository contains the implementation for the NeurIPS submission: RoboCLIP:One Demonstration is Enough to Learn Robot Policies. Please do not share/save this repository. An official version will be released upon acceptance. To use, please first download the `s3dg.py` file from `https://github.com/antoine77340/S3D_HowTo100M`, the official repo for our backbone model. Please also follow the instructions for downloading the `model_dict` and weights from the aforementioned repo. 

## Setting up the env

We recommend using conda for installation and provide a `.yml` file for installation. 

```sh
conda env create -f environment_roboclip.yml
```


## How To use it ?

To run experiments on the Metaworld environment suite with the sparse learnt reward, we need to first define what the demonstration to be used is. For textual input, uncomment line 225 and comment 226 and add the string prompt you would like to use in the `text_string` param. Similarly, if you would like to use human demonstration, uncomment line 226 and pass the path of the gif of the demonstration you would like to use. Similarly, for a metaworld video demo, set `human=False` and set the `video_path`. 

We provide the gifs used in our experiments within the `src/gifs`.
Then run: 
```sh
python metaworld_envs.py --env-type sparse_learnt --env-id drawer-open-v2-goal-hidden --dir-add <add experiment identifier>
```

To run the Kitchen experiments, similarly specify the gif path on line 345 and then run the following line with `--env-id` as `Kettle`, `Hinge` or `Slide`. 

```sh
python kitchen_env_wrappers.py --env-type sparse_learnt --env-id Kettle --dir-add <add experiment identifier>
```

These runs should produce default tensorboard experiments which save the best eval policy obtained by training on the RoboCLIP reward to disk. The plots in the paper are visualized by finetuning these policies for a handful of episodes. To replicate the Metaworld finetuning,  run:

```sh
python metaworld_envs.py --env-type dense_original --env-id drawer-open-v2-goal-hidden --pretrained <path_to_best_policy> --dir-add <add_experiment_identifier>  
```
