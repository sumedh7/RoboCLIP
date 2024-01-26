from gym import Env, spaces
import numpy as np
from stable_baselines3 import PPO
from d4rl_alt.kitchen.kitchen_envs import KitchenMicrowaveHingeSlideV0, KitchenKettleV0, KitchenSlideV0, KitchenHingeV0, KitchenMicrowaveV0, KitchenLightV0
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

def readGif(filename, asNumpy=True):
    """ readGif(filename, asNumpy=True)
    
    Read images from an animated GIF file.  Returns a list of numpy 
    arrays, or, if asNumpy is false, a list if PIL images.
    
    """
    
    # Check PIL
    if PIL is None:
        raise RuntimeError("Need PIL to read animated gif files.")
    
    # Check Numpy
    if np is None:
        raise RuntimeError("Need Numpy to read animated gif files.")
    
    # Check whether it exists
    if not os.path.isfile(filename):
        raise IOError('File not found: '+str(filename))
    
    # Load file using PIL
    pilIm = PIL.Image.open(filename)    
    pilIm.seek(0)
    
    # Read all images inside
    images = []
    try:
        while True:
            # Get image as numpy array
            tmp = pilIm.convert() # Make without palette
            a = np.asarray(tmp)
            if len(a.shape)==0:
                raise MemoryError("Too little memory to convert PIL image to array")
            # Store, and next
            images.append(a)
            pilIm.seek(pilIm.tell()+1)
    except EOFError:
        pass
    
    # Convert to normal PIL images if needed
    if not asNumpy:
        images2 = images
        images = []
        for im in images2:            
            images.append( PIL.Image.fromarray(im) )
    
    # Done
    return images

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--algo', type=str, default='ppo')
    parser.add_argument('--text-string', type=str, default='robot opening sliding door')
    parser.add_argument('--env-id', type=str, default='AssaultBullet-v0')
    parser.add_argument('--env-type', type=str, default='AssaultBullet-v0')
    parser.add_argument('--total-time-steps', type=int, default=5000000)
    parser.add_argument('--n-envs', type=int, default=8)
    parser.add_argument('--n-steps', type=int, default=128)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--dir-add', type=str, default='')


    args = parser.parse_args()
    return args


class KitchenEnvSlidingReward(Env):
    def __init__(self, text_string):
        super(KitchenEnvSlidingReward,self)
        env = KitchenMicrowaveHingeSlideV0()
        self.env = TimeLimit(env, max_episode_steps=128)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.past_observations = []
        self.window_length = 16
        self.net = S3D('s3d_dict.npy', 512)

        # Load the model weights
        self.net.load_state_dict(th.load('s3d_howto100m.pth'))
        # Evaluation mode
        self.net = self.net.eval()
        text_output = self.net.text_module([text_string])
        self.target_embedding = text_output['text_embedding']

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)


    def preprocess_kitchen(self, frames):
        frames = np.array(frames)
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        # frames = frames[:, :,::4,:,:]
        return frames
        
    
    def render(self, mode="rgb_array", width=250, height=250):
        return self.env.render(mode=mode, width=width, height=height)


    def step(self, action):
        

        obs, _, done, info = self.env.step(action)
        if done:
            return obs, 0.0, done, info
        self.past_observations.append(self.env.render(mode="rgb_array", width=250, height=250))
        self.past_observations = self.past_observations[-self.window_length:]
        if len(self.past_observations)<self.window_length:
            reward=0.0
        else:
            frames = self.preprocess_kitchen(self.past_observations)

            video = th.from_numpy(frames)
            # print(frames.shape)
            video_output = self.net(video.float())

            video_embedding = video_output['video_embedding']
            similarity_matrix = th.matmul(self.target_embedding, video_embedding.t())

            reward = similarity_matrix.detach().numpy()[0][0]
        
        

        return obs, reward, done, info

    def reset(self):
        self.past_observations = []
        return self.env.reset()



class KitchenEnvSparseReward(Env):
    def __init__(self, text_string=None, time=False, video_path=None):
        super(KitchenEnvSparseReward,self)
        env = KitchenMicrowaveHingeSlideV0()
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
        if text_string:
            text_output = self.net.text_module([text_string])
            self.target_embedding = text_output['text_embedding']
        if video_path:
            frames = readGif(video_path)
            frames = self.preprocess_kitchen(frames)
            video = th.from_numpy(frames)
            video_output = self.net(video.float())
            self.target_embedding = video_output['video_embedding']
        assert self.target_embedding is not None

        self.counter = 0

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)


    def preprocess_kitchen(self, frames):
        frames = np.array(frames)
        frames = frames[None, :,:,:,:]
        frames = frames.transpose(0, 4, 1, 2, 3)
        frames = frames[:, :,::4,:,:]
        if np.equal(np.mod(frames, 1).all(), 0):
            frames = frames/255
        return frames
        
    
    def render(self, mode="rgb_array", width=250, height=250):
        return self.env.render(mode=mode, width=width, height=height)


    def step(self, action):
        obs, _, done, info = self.env.step(action)
        self.past_observations.append(self.env.render(mode="rgb_array", width=250, height=250))
        t = info["time"]
        if self.time:
            obs = np.concatenate([obs, np.array([t])])
        if done:
            frames = self.preprocess_kitchen(self.past_observations)
            
        
        
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


class KitchenEnvSparseOriginalReward(Env):
    def __init__(self, time=False):
        super(KitchenEnvSparseOriginalReward, self)
        env = KitchenSlideV0()
        self.env = TimeLimit(env, max_episode_steps=128)
        self.time = time
        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0]+1,), dtype=np.float32)
        self.action_space = self.env.action_space
        self.episode_reward = 0.0

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)
    
    def render(self, mode="rgb_array", width=250, height=250):
        return self.env.render(mode=mode, width=width, height=height)

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        t = info["time"]
        if self.time:
            obs = np.concatenate([obs, np.array([t])])
        self.episode_reward += r
        # self.past_observations.append(self.env.render(mode="rgb_array", width=250, height=250))
        if done:
            return obs, self.episode_reward, done, info
        return obs, 0.0, done, info

    def reset(self):
        self.episode_reward = 0.0
        # self.past_observations = []
        if not self.time:
            return self.env.reset()
        return np.concatenate([self.env.reset(), np.array([0.0])]) 

class KitchenEnvDenseOriginalReward(Env):
    def __init__(self, env_id, time=False):
        super(KitchenEnvDenseOriginalReward, self)
        if env_id=="Hinge":
            env = KitchenHingeV0()
        elif env_id=="Microwave":
            env = KitchenMicrowaveV0()
        elif env_id=="Slide":
            env = KitchenSlideV0()
        elif env_id=="Kettle":
            env = KitchenKettleV0()
        else:
            raise Exception("Unknown env_id")
        self.env = TimeLimit(env, max_episode_steps=128)
        self.time = time
        if not self.time:
            self.observation_space = self.env.observation_space
        else:
            self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0]+1,), dtype=np.float32)
        self.action_space = self.env.action_space
        self.episode_reward = 0.0

    def get_obs(self):
        return self.baseEnv._get_obs(self.baseEnv.prev_time_step)
    
    def render(self, mode="rgb_array", width=250, height=250):
        return self.env.render(mode=mode, width=width, height=height)

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        t = info["time"]
        if self.time:
            obs = np.concatenate([obs, np.array([t])])
        self.episode_reward += r
        # self.past_observations.append(self.env.render(mode="rgb_array", width=250, height=250))
        return obs, r, done, info

    def reset(self):
        self.episode_reward = 0.0
        # self.past_observations = []
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
        if env_type == "sparse_learnt":
            # env = KitchenEnvSparseReward(args.text_string, time=True)
            env = KitchenEnvSparseReward(time=True, video_path="/lab/ssontakk/S3D_HowTo100M/cem_planning/gifs/Kettle_dense_original.gif")
        elif env_type == "dense_learnt":
            env = KitchenEnvSlidingReward(args.text_string)
        elif env_type == "sparse_original":
            env = KitchenEnvSparseOriginalReward(time=True)
        else:
            env = KitchenEnvDenseOriginalReward(env_id, time=True)
            env = TimeLimit(env, max_episode_steps=128)
        env = Monitor(env, os.path.join(log_dir, str(rank)))
        env.seed(seed + rank)
        return env
    # set_global_seeds(seed)
    return _init


def main(j):
    global args
    global log_dir
    args = get_args()
    log_dir = f"kitchen/{args.env_id}_{args.env_type}{args.dir_add}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    envs = SubprocVecEnv([make_env(args.env_type, args.env_id, i) for i in range(args.n_envs)])


    # model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log=log_dir, n_steps=128, batch_size=32, n_epochs=20, ent_coef=1.2286936854345043e-07, 
    #             gamma=0.95, learning_rate=0.028922505128289214,
    #             clip_range=0.1, gae_lambda=1.0, max_grad_norm=1, vf_coef=0.8169885733544515)
    if not args.pretrained:
        if args.env_id == "Microwave":
            model = PPO("MlpPolicy", envs, verbose=1, batch_size=16, n_steps=512, gamma=0.9, learning_rate=0.12073464649899267, ent_coef=1.0882514144866191e-07, clip_range=0.2, 
                        n_epochs=5, gae_lambda=0.92, max_grad_norm=1, vf_coef=0.737910749115419, seed=j)

        


        # elif args.env_id == "Kettle":
        #     model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log=log_dir, batch_size=8, n_steps=32,  gamma=0.999, learning_rate=0.0002533328865146111, 
        #                 ent_coef=5.835533552192642e-07, clip_range= 0.4, n_epochs=5, gae_lambda=0.95, max_grad_norm=0.7, vf_coef=0.7296829645393675,)
        


        else:
            model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log=log_dir, n_steps=args.n_steps, batch_size=args.n_steps*args.n_envs, n_epochs=1, ent_coef=0.5)
    else:
        model = PPO.load(args.pretrained, env=envs, tensorboard_log=log_dir)
    # eval_env = KitchenSlideV0()#KitchenEnvSparseOriginalReward(time=False)

    eval_env = SubprocVecEnv([make_env("dense_original", args.env_id, j) for i in range(1)])#KitchenEnvDenseOriginalReward(args.env_id, time=True)
    # eval_env = TimeLimit(eval_env, max_episode_steps=128)
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=500,
                                 deterministic=True, render=False)

    model.learn(total_timesteps=int(args.total_time_steps), callback=eval_callback)
    model.save(f"{log_dir}/trained")

if __name__ == '__main__':
    main(11)