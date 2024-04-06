from warnings import warn
from pdb import set_trace
from multiprocessing import Process, Queue
from functools import partial

from torch._C import device
from play import PyGymCallback, Player,EvalCallback

import gym
import pygame
import os 
import numpy as np
import time 
import datetime as dt
from itertools import count 
from typing import Tuple
from collections import deque 

import scipy
import matplotlib.pyplot as plt
from scipy.stats import uniform, gamma, norm, exponnorm
import matplotlib.pyplot as plt 

import torch
import torch.nn.functional as F
import torchvision.transforms as T 
import torch.optim as optim
from torch import nn
from gym import Env, spaces
import numpy as np
from stable_baselines3 import PPO
import torch as th
from s3dg import S3D
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
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

import metaworld
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

from kitchen_env_wrappers import readGif
from matplotlib import animation
import matplotlib.pyplot as plt

torch.manual_seed(10)

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

class MetaworldInteractive(Env):
    def __init__(self, env_id, text_string=None, time=False, video_path=None, rank=0, human=True):
        super(MetaworldInteractive,self)
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
        if text_string:
            text_output = self.net.text_module([text_string])
            self.target_embedding = text_output['text_embedding']
        if video_path:
            frames = readGif(video_path)
            
            if human:
                frames = self.preprocess_human_demo(frames)
            else:
                frames = self.preprocess_metaworld(frames)
            if frames.shape[1]>3:
                frames = frames[:,:3]
            video = th.from_numpy(frames)
            video_output = self.net(video.float())
            self.target_embedding = video_output['video_embedding']
        assert self.target_embedding is not None

        self.counter = 0

    def get_obs(self):
        return self.env._get_obs(self.env.prev_time_step)
    
    def get_counter(self):
        return self.counter
    
    def show_progress(self):
        for i in range(3):
                for frame in self.past_observations: 
                    cv2.imshow('Frame', frame)
                    cv2.waitKey(100)  # 100ms delay between frames
                cv2.destroyAllWindows()  # Close the window after displaying the frames
    
    def get_similarity(self,feedback,shorten=False):
        text_output = self.net.text_module([feedback])
        self.target_embedding = text_output['text_embedding']
        frames = self.preprocess_metaworld(self.past_observations,shorten=shorten)
        video = th.from_numpy(frames)
        video_output = self.net(video.float())
        video_embedding = video_output['video_embedding']
        similarity_matrix = th.matmul(self.target_embedding, video_embedding.t())
        reward = similarity_matrix.detach().numpy()[0][0]
        return reward

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
        obs, _, done, info = self.env.step(action.detach().cpu().numpy())
        self.past_observations.append(self.env.render())
        self.counter += 1
        t = self.counter/128
        if self.time:
            obs = np.concatenate([obs, np.array([t])])
        if done:
            frames = self.preprocess_metaworld(self.past_observations,shorten=True)
            video = th.from_numpy(frames)
            video_output = self.net(video.float())
            video_embedding = video_output['video_embedding']
            similarity_matrix = th.matmul(self.target_embedding, video_embedding.t())
            reward = similarity_matrix.detach().numpy()[0][0]
        else:
            reward = 0.0  # Default reward for steps without feedback
        return obs, reward, done, info

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

class Head(nn.Module):
    def __init__(self,Input_dims,Output_dims):
        super(Head,self).__init__()

        self.linear_1 = nn.Linear(Input_dims,16)
        self.linear_2 = nn.Linear(16,Output_dims)
    
    def forward(self, x):
        x = x.float()
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = torch.tanh(x)  # Action values are in [-1, 1]
        return x
        
class CreditAssignment():
    def __init__(self, dist: scipy.stats.rv_continuous):
        self.dist = dist
        
    def __call__(self, s_start: int, h_start: int) -> int:
        step_diff = abs(h_start - s_start)
        return self.dist.cdf(step_diff)
        
    def _normalize(self, s_start: float, s_end: float, h_start: float) -> Tuple[float, float]: 
        s_norm_start =  h_start - s_start
        s_norm_end = h_start - s_end
        return s_norm_start, s_norm_end
    
    def show_dist(self, s_start: float, s_end: float, h_start: float):
        s_norm_start, s_norm_end = self._normalize(s_start, s_end, h_start)
        x = np.linspace(self.dist.ppf(.01), self.dist.ppf(.99))
        plt.plot(x, self.dist.pdf(x), 'r-')
        plt.vlines(s_norm_start,ymin=0, ymax=self.dist.pdf(s_norm_start), color='green')
        plt.vlines(s_norm_end, ymin=0, ymax=self.dist.pdf(s_norm_end), color='green')

class BufferDeque():
    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, index):
        if isinstance(index, (tuple, list)):
          pointer = list(self.memory)
          return [pointer[i] for i in index]
        return list(self.memory[index])

    def push(self, tensor):
        self.memory.append(tensor) 

    def random_sample(self, batch_size):
        rand_idx = np.random.randint(len(self.memory),  size=batch_size)
        rand_batch = [self.memory[i] for i in rand_idx]
        state, action, feedback, credit =  [], [], [], []
        for s, a, f, c in rand_batch:
            state.append(s)
            action.append(a)
            feedback.append(f)
            credit.append(c)

        return torch.cat(state), torch.tensor(action), torch.tensor(feedback), torch.tensor(credit)

class NetworkController(PyGymCallback):
    '''
    state_start_time :- state start time is collected before step in before_step()
    state_end_time :- state end time is collected after the step in after_step()
    h_time :- time at which the feedback is recorded, which comes along with the feedback in a list in after_step function
    
    questions:-
        1. why random sample the credit?
        2. Are my assumptons about start and end of state time correct?
        3. Why perform the backward function in both after_set_action() and after_set() [as per psuedo code] isn't
            performing it in after_set_action() enough ?
        4. Not able to get the torch.stack(credit) to work in buffer.random_sample() ? 

    Yet-To-Do:-

    adding backward function
    '''
    def __init__(self, head, queue, img_dims = (3, 160, 160), ts_len = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.head = head
        self.queue = queue 
        self.img_dims = img_dims
        self.ts_len = ts_len
        self.dims = 10000
        self.buffer = BufferDeque(self.dims)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sliding_window = deque()
        self.opt = optim.Adam(list(self.head.parameters()), lr=1e-1, weight_decay = 1e-1 )

    def backward(self, state, action, feedback, credit):
        state = state.to(self.device)
        credit = credit.to(self.device)
        feedback = feedback.to(self.device)
        action = action.to(self.device)

        self.loss_list = []
        h_hat = self.head(state)
        h_hat_s_a = h_hat[:, action]
        # h_hat_s_a.requires_grad = True
        L = torch.mean(credit*(h_hat_s_a - feedback)**2)
        self.opt.zero_grad() 
        L.backward()
        self.opt.step()
        self.loss_list.append(L)
        print("backward")
        # print(f"feedback : {feedback}")
        # print(f"loss: {L}, q_values: {h_hat}")

        def before_step(self):
            fb=0
            print("STEP: ",self.env.get_counter())
            if self.env.get_counter() % 32 == 0 and self.env.get_counter()!=0:
                print("32 steps reached. Please provide feedback:")
                self.env.show_progress()
                new_instruction = input()  # Assuming new instruction is text. Modify as needed.
                # Process new instruction
                fb = self.env.get_similarity(new_instruction,False)
                print("Feedback: ",fb)
            self.state_start_time = self.env.get_counter()
            time.sleep(5e-3)
            self.queue.put(
                    dict(
                        feedback = fb,
                        h_time = self.env.get_counter()
                    ))
            

        def before_set_action(self):
            pass 
            
        
        def set_action(self):
            self.network_output = self.head(self.play.state)
            # self.play.action = np.argmax(self.network_output.cpu().detach().numpy())
            self.play.action = self.network_output#.cpu().detach().numpy())

        def after_set_action(self):
            batch=64
            loss_fn = nn.MSELoss(reduction = 'mean') 
            #only when buffer has 50 feedbacks
        
            if len(self.buffer) > 50:      
                # Only train every certain number of steps
                if self.t % 16 == 0: 
                    #rand_batch = np.random.randint(len(self.buffer), size=batch_size) 
                    state, action, feedback, credit = self.buffer.random_sample(batch)
                    self.backward( state, action, feedback, credit) 
        
        def after_step(self):
            fb_dict = self.queue.get()
            # fb = torch.tensor(fb_dict["feedback"]).to(self.device)
            fb = fb_dict['feedback']
            h_time = fb_dict["h_time"]
            self.sliding_window.append(
                dict(
                    state = self.state, 
                    action = self.action, 
                    feedback = fb, 
                    s_start = self.state_start_time, 
                )
            )

            if fb != 0:
                ca = CreditAssignment(norm(loc=0, scale=1))
                state, action, credit = [], [], []
                for win in self.sliding_window:
                    credit_for_state = ca(s_start=win["s_start"],h_start=h_time)
                    print("credit",credit_for_state,"s_start",win["s_start"],"h_start",h_time)
                    if credit_for_state !=0:
                        print("credit!=0")
                        state.append(win['state'])
                        action.append(win['action'])
                        # credit.append(torch.tensor(credit_for_state, dtype=torch.float32).to(self.device))
                        credit.append(credit_for_state) 
                        self.buffer.push([
                            win['state'], 
                            win['action'],
                            fb,
                            credit_for_state
                            # torch.Tensor(credit) 
                        ])
                #print("state",state,"action",action,"credit",credit)
                state, action, credit = torch.cat(state), torch.stack(action), torch.tensor(credit)
                feedback = torch.full(credit.size(), fb)
                self.backward( state, action, feedback, credit)

    def after_play(self):
        plt.title('Head_Network_Error')
        plt.plot(self.loss_list)
        plt.savefig('Test_Error')


def make_env(env_type, env_id, text_string,rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    # env = KitchenMicrowaveHingeSlideV0()
    if env_type=="interactive":
        env= MetaworldInteractive(env_id=env_id, text_string=text_string, time=True, rank=rank, human=True)
    else:
        env = MetaworldDense(env_id=env_id, time=True, rank=rank)
    #env = Monitor(env, os.path.join(log_dir, str(rank)))
    # env.seed(seed + rank)
    return env
        

def main():
    global args
    global log_dir
    args = get_args()

    env = make_env(args.env_type, args.env_id,args.text_string, 0)
    #env=MetaworldInteractive(env_id=args.env_id, text_string=args.text_string, time=True, rank=0, human=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_env=make_env("dense_original", args.env_id,None, 0)
    log_dir = f"metaworld/{args.env_id}_{args.env_type}{args.dir_add}"
        # Assuming 'env' is your Gym environment instance
    '''observation_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        print("Discrete")
        # For discrete action spaces
        action_dim = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        print("Box")
        # For continuous action spaces
        action_dim = env.action_space.shape[0]
        action_space_low = env.action_space.low
        action_space_high = env.action_space.high
    else:
        raise NotImplementedError("Action space type not supported")

    print("Observation dimension:", observation_dim)
    print("Action dimension:", action_dim)

    print("Action space low:", action_space_low)
    print("Action space high:", action_space_high)
    pr'''

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    

    head_net = Head(40,4).to(dtype=torch.float32)
    head_net = head_net.to(device)
    
    
    # opt = torch.optim.Adam(head_net.parameters(), lr=1e-4, weight_decay=1e-1) 
    eval_callback = EvalCallback(
        eval_env=eval_env,
        model=head_net,
        save_path=log_dir,
        eval_freq=500,
        verbose=1
    )
    
    Feedback_queue = Queue()

    player = Player(callbacks=[NetworkController(head=head_net, queue=Feedback_queue,
                                                    env=env, human=True),eval_callback]) #pass the queue
    player.play(n_episodes=args.total_time_steps/128, n_steps = args.total_time_steps)
    

if __name__ == "__main__":
    main() 