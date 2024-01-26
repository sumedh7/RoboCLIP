from gym import Env, spaces
import numpy as np
from stable_baselines3 import PPO
# from d4rl_alt.kitchen.kitchen_envs import KitchenMicrowaveHingeSlideV0, KitchenKettleV0
import torch as th
from s3dg import S3D
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from PIL import Image, ImageSequence
import torch as th
from s3dg import S3D
import numpy as np
from PIL import Image, ImageSequence
import cv2
import gif2numpy
import PIL
import os
import seaborn as sns
import matplotlib.pylab as plt

from typing import Any, Dict

import gym
from gym.spaces import Box
import torch as th

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
import os
from stable_baselines3.common.monitor import Monitor
from memory_profiler import profile
import argparse
from stable_baselines3.common.callbacks import EvalCallback

import metaworld
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

from kitchen_env_wrappers import readGif
from matplotlib import animation
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--text-string', type=str, default='robot opening sliding door')
    parser.add_argument('--dir-add', type=str, default='')
    parser.add_argument('--env-id', type=str, default='AssaultBullet-v0')
    parser.add_argument('--env-type', type=str, default='AssaultBullet-v0')
    parser.add_argument('--total-time-steps', type=int, default=5000000)
    parser.add_argument('--n-envs', type=int, default=8)
    parser.add_argument('--n-steps', type=int, default=128)
    parser.add_argument('--pretrained', type=str, default=None)


    args = parser.parse_args()
    return args
class MetaworldSparseDual(Env):
    def __init__(self, env_id, text_string=None, time=False, video_path=None, rank=0, human=True):
        super(MetaworldSparseDual,self)
        door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id]
        env = door_open_goal_hidden_cls(seed=rank)
        self.env = TimeLimit(env, max_episode_steps=128)
        self.time = time
        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0]+1,), dtype=np.float32)
        self.action_space = self.env.action_space
        self.past_observations = []
        self.window_length = 16
        self.net = S3D('s3d_dict.npy', 512)

        # Load the model weights
        self.net.load_state_dict(th.load('s3d_howto100m.pth'))
        # Evaluation mode
        self.net = self.net.eval()
        # self.target_embedding = None
        if text_string:
            text_output = self.net.text_module([text_string])
            self.target_text_embedding = text_output['text_embedding']
        if video_path:
            frames = readGif(video_path)
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$", len(frames))
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$", frames[0].shape)
            if human:
                frames = self.preprocess_human_demo(frames)
            else:
                frames = self.preprocess_metaworld(frames)
            if frames.shape[1]>3:
                frames = frames[:,:3]
            print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$", frames.shape)
            video = th.from_numpy(frames)
            video_output = self.net(video.float())
            self.target_video_embedding = video_output['video_embedding']
        

        self.counter = 0

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)

    def preprocess_human_demo(self, frames):
        frames = np.array(frames)
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        return frames

    def preprocess_metaworld(self, frames, shorten=True):
        center = 240, 320
        h, w = (250, 250)
        x = int(center[1] - w/2)
        y = int(center[0] - h/2)
        # frames = np.array([cv2.resize(frame, dsize=(250, 250), interpolation=cv2.INTER_CUBIC) for frame in frames])
        frames = np.array([frame[y:y+h, x:x+w] for frame in frames])
        a = frames
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        if shorten:
            frames = frames[:, :,::4,:,:]
        # frames = frames/255
        return frames
        
    
    def render(self):
        frame = self.env.render()
        # center = 240, 320
        # h, w = (250, 250)
        # x = int(center[1] - w/2)
        # y = int(center[0] - h/2)
        # frame = frame[y:y+h, x:x+w]
        return frame


    def step(self, action):
        obs, _, done, info = self.env.step(action)
        self.past_observations.append(self.env.render())
        self.counter += 1
        t = self.counter/128
        if self.time:
            obs = np.concatenate([obs, np.array([t])])
        if done:
            frames = self.preprocess_metaworld(self.past_observations)
            
        
        
            video = th.from_numpy(frames)
            # print("video.shape", video.shape)
            # print(frames.shape)
            video_output = self.net(video.float())

            video_embedding = video_output['video_embedding']
            reward = 0.0
            similarity_matrix_video = th.matmul(self.target_video_embedding, video_embedding.t())

            reward += similarity_matrix_video.detach().numpy()[0][0]
            similarity_matrix_text = th.matmul(self.target_text_embedding, video_embedding.t())
            reward += similarity_matrix_text.detach().numpy()[0][0]
            return obs, reward, done, info
        return obs, 0.0, done, info

    def reset(self):
        self.past_observations = []
        self.counter = 0
        if not self.time:
            return self.env.reset()
        return np.concatenate([self.env.reset(), np.array([0.0])])


class MetaworldSparseEdit(Env):
    def __init__(self, env_id, text_string=None, time=False, video_path=None, rank=0, human=True):
        super(MetaworldSparseEdit,self)
        door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id]
        env = door_open_goal_hidden_cls(seed=rank)
        self.env = TimeLimit(env, max_episode_steps=128)
        self.time = time
        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0]+1,), dtype=np.float32)
        self.action_space = self.env.action_space
        self.past_observations = []
        self.window_length = 16
        self.net = S3D('s3d_dict.npy', 512)

        # Load the model weights
        self.net.load_state_dict(th.load('s3d_howto100m.pth'))
        # Evaluation mode
        self.net = self.net.eval()
        self.target_embedding = None
        # if text_string:
        #     text_output = self.net.text_module([text_string])
        #     self.target_embedding = text_output['text_embedding']
        frames = readGif(video_path)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$", len(frames))
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$", frames[0].shape)
        if human:
            frames = self.preprocess_human_demo(frames)
        else:
            frames = self.preprocess_metaworld(frames)
        if frames.shape[1]>3:
            frames = frames[:,:3]
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$", frames.shape)
        video = th.from_numpy(frames)
        video_output = self.net(video.float())
        self.video_emb = video_output['video_embedding']

        text_output = self.net.text_module(["black box"])
        red_button_embedding = text_output['text_embedding']

        text_output = self.net.text_module(["green drawer"])
        green_drawer_embedding = text_output['text_embedding']

        self.target_embedding = self.video_emb - red_button_embedding + green_drawer_embedding

        assert self.target_embedding is not None

        self.counter = 0

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)

    def preprocess_human_demo(self, frames):
        frames = np.array(frames)
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        return frames

    def preprocess_metaworld(self, frames, shorten=True):
        center = 240, 320
        h, w = (250, 250)
        x = int(center[1] - w/2)
        y = int(center[0] - h/2)
        # frames = np.array([cv2.resize(frame, dsize=(250, 250), interpolation=cv2.INTER_CUBIC) for frame in frames])
        frames = np.array([frame[y:y+h, x:x+w] for frame in frames])
        a = frames
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        if shorten:
            frames = frames[:, :,::4,:,:]
        # frames = frames/255
        return frames
        
    
    def render(self):
        frame = self.env.render()
        # center = 240, 320
        # h, w = (250, 250)
        # x = int(center[1] - w/2)
        # y = int(center[0] - h/2)
        # frame = frame[y:y+h, x:x+w]
        return frame


    def step(self, action):
        obs, _, done, info = self.env.step(action)
        self.past_observations.append(self.env.render())
        self.counter += 1
        t = self.counter/128
        if self.time:
            obs = np.concatenate([obs, np.array([t])])
        if done:
            frames = self.preprocess_metaworld(self.past_observations)
            
        
        
            video = th.from_numpy(frames)
            # print("video.shape", video.shape)
            # print(frames.shape)
            video_output = self.net(video.float())

            video_embedding = video_output['video_embedding']
            similarity_matrix = th.matmul(self.target_embedding, video_embedding.t())

            reward = similarity_matrix.detach().numpy()[0][0]
            return obs, reward, done, info
        return obs, 0.0, done, info

    def reset(self):
        self.past_observations = []
        self.counter = 0
        if not self.time:
            return self.env.reset()
        return np.concatenate([self.env.reset(), np.array([0.0])])



class MetaworldSparseCLIP(Env):
    def __init__(self, env_id, text_string=None, time=False, video_path=None, rank=0, human=True):
        super(MetaworldSparseCLIP,self)
        door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id]
        env = door_open_goal_hidden_cls(seed=rank)
        self.env = TimeLimit(env, max_episode_steps=128)
        self.time = time
        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0]+1,), dtype=np.float32)
        self.action_space = self.env.action_space
        self.past_observations = []
        self.window_length = 16
        self.text_string = text_string
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)

    def preprocess_human_demo(self, frames):
        frames = np.array(frames)
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        return frames

    def preprocess_metaworld(self, frames, shorten=True):
        center = 240, 320
        h, w = (250, 250)
        x = int(center[1] - w/2)
        y = int(center[0] - h/2)
        # frames = np.array([cv2.resize(frame, dsize=(250, 250), interpolation=cv2.INTER_CUBIC) for frame in frames])
        frames = np.array([frame[y:y+h, x:x+w] for frame in frames])
        a = frames
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        if shorten:
            frames = frames[:, :,::4,:,:]
        # frames = frames/255
        return frames
        
    
    def render(self):
        frame = self.env.render()
        # center = 240, 320
        # h, w = (250, 250)
        # x = int(center[1] - w/2)
        # y = int(center[0] - h/2)
        # frame = frame[y:y+h, x:x+w]
        return frame


    def step(self, action):
        obs, _, done, info = self.env.step(action)
        self.past_observations.append(self.env.render())
        self.counter += 1
        t = self.counter/128
        if self.time:
            obs = np.concatenate([obs, np.array([t])])
        if done:
            frame = self.past_observations[-1]
            inputs = self.processor(text=[self.text_string], images=frame, return_tensors="pt", padding=True)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image

            reward = logits_per_image.detach().numpy()[0][0]
            return obs, reward, done, info
        return obs, 0.0, done, info

    def reset(self):
        self.past_observations = []
        self.counter = 0
        if not self.time:
            return self.env.reset()
        return np.concatenate([self.env.reset(), np.array([0.0])])


class MetaworldSparseMulti(Env):
    def __init__(self, env_id, text_string=None, time=False, video_path=None, rank=0, human=True, num_demo=1):
        super(MetaworldSparseMulti,self)
        self.num_demo = num_demo
        door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id]
        env = door_open_goal_hidden_cls(seed=rank)
        self.env = TimeLimit(env, max_episode_steps=128)
        self.time = time
        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0]+1,), dtype=np.float32)
        self.action_space = self.env.action_space
        self.past_observations = []
        self.window_length = 16
        self.net = S3D('s3d_dict.npy', 512)

        # Load the model weights
        # self.net.load_state_dict(th.load('s3d_howto100m.pth'))
        # Evaluation mode
        self.net = self.net.eval()
        self.targets = []
        if video_path:
            for i in range(num_demo):
                path = video_path+f"/button-press-v2_{i}.gif"
                frames = readGif(path)
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$", len(frames))
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$", frames[0].shape)
                if human:
                    frames = self.preprocess_human_demo(frames)
                else:
                    frames = self.preprocess_metaworld(frames)
                if frames.shape[1]>3:
                    frames = frames[:,:3]
                print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$", frames.shape)
                video = th.from_numpy(frames)
                video_output = self.net(video.float())
                self.targets.append(video_output['video_embedding'])

        self.counter = 0

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)

    def preprocess_human_demo(self, frames):
        frames = np.array(frames)
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        return frames

    def preprocess_metaworld(self, frames, shorten=True):
        center = 240, 320
        h, w = (250, 250)
        x = int(center[1] - w/2)
        y = int(center[0] - h/2)
        # frames = np.array([cv2.resize(frame, dsize=(250, 250), interpolation=cv2.INTER_CUBIC) for frame in frames])
        frames = np.array([frame[y:y+h, x:x+w] for frame in frames])
        a = frames
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        if shorten:
            frames = frames[:, :,::4,:,:]
        # frames = frames/255
        return frames
        
    
    def render(self):
        frame = self.env.render()
        # center = 240, 320
        # h, w = (250, 250)
        # x = int(center[1] - w/2)
        # y = int(center[0] - h/2)
        # frame = frame[y:y+h, x:x+w]
        return frame


    def step(self, action):
        obs, _, done, info = self.env.step(action)
        self.past_observations.append(self.env.render())
        self.counter += 1
        t = self.counter/128
        if self.time:
            obs = np.concatenate([obs, np.array([t])])
        if done:
            frames = self.preprocess_metaworld(self.past_observations)
            
        
        
            video = th.from_numpy(frames)
            # print("video.shape", video.shape)
            # print(frames.shape)
            video_output = self.net(video.float())

            video_embedding = video_output['video_embedding']
            reward = 0.0
            for i in range(self.num_demo):
                similarity_matrix = th.matmul(self.targets[i], video_embedding.t())
                reward += similarity_matrix.detach().numpy()[0][0]
            return obs, reward, done, info
        return obs, 0.0, done, info

    def reset(self):
        self.past_observations = []
        self.counter = 0
        if not self.time:
            return self.env.reset()
        return np.concatenate([self.env.reset(), np.array([0.0])])

class MetaworldDense(Env):
    def __init__(self, env_id, time=False, rank=0):
        super(MetaworldDense,self)
        door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id]
        env = door_open_goal_hidden_cls(seed=rank)
        self.env = TimeLimit(env, max_episode_steps=128)
        self.time = time
        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0]+1,), dtype=np.float32)
        self.action_space = self.env.action_space
        self.past_observations = []
        
        self.counter = 0

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)
        
    
    def render(self):
        # camera_name="topview"
        frame = self.env.render()
        return frame


    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # self.past_observations.append(self.env.render())
        self.counter += 1
        t = self.counter/128
        if self.time:
            obs = np.concatenate([obs, np.array([t])])
        return obs, reward, done, info
        
    def reset(self):
        self.counter = 0
        if not self.time:
            return self.env.reset()
        return np.concatenate([self.env.reset(), np.array([0.0])])



def make_env(env_type, env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        # env = KitchenMicrowaveHingeSlideV0()
        if env_type == "sparse_learnt_dual":
            # env = MetaworldSparse(env_id=env_id, text_string="robot closing green drawer", time=True, rank=rank)
            env = MetaworldSparseDual(env_id=env_id, video_path="/lab/ssontakk/S3D_HowTo100M/cem_planning/gifs/button-press-v2.gif", text_string="robot pushing red button", 
                time=True, rank=rank, human=False)
        
        elif env_type == "sparse_learnt_edit":
            env = MetaworldSparseEdit(env_id=env_id, video_path="/lab/ssontakk/S3D_HowTo100M/cem_planning/gifs/door-open-v2.gif", 
                time=True, rank=rank, human=False)
        elif env_type == "sparse_learnt_clip":
            env = MetaworldSparseCLIP(env_id=env_id, text_string="robot pushing red button", 
                time=True, rank=rank, human=False)
        elif env_type == "sparse_learnt_multi":
            env = MetaworldSparseMulti(env_id=env_id, video_path="/lab/ssontakk/S3D_HowTo100M/cem_planning/gifs/multi_demo", 
                time=True, rank=rank, human=False, num_demo=1)
        else:
            env = MetaworldDense(env_id=env_id, time=True, rank=rank)
        env = Monitor(env, os.path.join(log_dir, str(rank)))
        # env.seed(seed + rank)
        return env
    # set_global_seeds(seed)
    return _init



def main():
    global args
    global log_dir
    args = get_args()
    log_dir = f"metaworld/{args.env_id}_{args.env_type}{args.dir_add}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    envs = SubprocVecEnv([make_env(args.env_type, args.env_id, i) for i in range(args.n_envs)])

    if not args.pretrained:
        model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log=log_dir, n_steps=args.n_steps, batch_size=args.n_steps*args.n_envs, n_epochs=1, ent_coef=0.5)
    else:
        model = PPO.load(args.pretrained, env=envs, tensorboard_log=log_dir)

    eval_env = SubprocVecEnv([make_env("dense_original", args.env_id, i) for i in range(10, 10+args.n_envs)])#KitchenEnvDenseOriginalReward(time=True)

    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=500,
                                 deterministic=True, render=False)

    model.learn(total_timesteps=int(args.total_time_steps), callback=eval_callback)
    model.save(f"{log_dir}/trained")



if __name__ == '__main__':
    main()
