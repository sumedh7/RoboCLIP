from warnings import warn
from pdb import set_trace
from multiprocessing import Process, Queue
from functools import partial

from torch._C import device
from torch.utils.tensorboard import SummaryWriter
import pygame
import os 
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
from stable_baselines3 import PPO
import torch as th
from s3dg import S3D
from gym.wrappers.time_limit import TimeLimit
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
	parser.add_argument('--timed', type=int, default=1)
	parser.add_argument('--VLM', type=int, default=1)
	parser.add_argument('--n-frames', type=int, default=32)
	parser.add_argument('--evaluative', type=int, default=1)
	parser.add_argument('--bounded',type=int,default=0)
	args = parser.parse_args()
	return args

def mish(input):
	return input * torch.tanh(F.softplus(input))

class Mish(nn.Module):
	def __init__(self): super().__init__()
	def forward(self, input): return mish(input)

class MetaworldInteractive(Env):
	def __init__(self, env_id, max_episode_steps,text_string=None, time=False, video_path=None, rank=0, human=True):
		super(MetaworldInteractive,self)
		door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id]
		env = door_open_goal_hidden_cls(seed=rank)
		if time:
			self.env = TimeLimit(env, max_episode_steps)
		else:
			self.env=env
		self.time = time
		if not self.time:
			self.observation_space = self.env.observation_space
		else:
			self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0]+1,), dtype=np.float32)
		self.action_space = self.env.action_space
		self.past_observations = []
		self.window_length = 16
		self.net = S3D('s3d_dict.npy', 512)
		self.max_episode_steps=max_episode_steps

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
		'''if len(self.past_observations)%self.max_episode_steps==0:
			frames=self.past_observations[-self.max_episode_steps:]
		else:
			frames=self.past_observations[-32:]'''
		frames=self.past_observations[-32:]
		for i in range(3):
				for frame in frames: 
					cv2.imshow('Frame', frame)
					cv2.waitKey(100)  # 100ms delay between frames
				cv2.destroyAllWindows()  # Close the window after displaying the frames

				'''center = 240, 320
				h, w = (250, 250)
				x = int(center[1] - w/2)
				y = int(center[0] - h/2)
				frames2 = np.array([frame[y:y+h, x:x+w] for frame in frames])
				for frame2 in frames2: 
					cv2.imshow('Frame', frame2)
					cv2.waitKey(100)  # 100ms delay between frames
				cv2.destroyAllWindows()  # Close the window after displaying the frames'''
	
	def get_similarity(self,feedback,shorten=False):
		text_output = self.net.text_module([feedback])
		self.target_embedding = text_output['text_embedding']
		frames = self.preprocess_metaworld(self.past_observations[-args.n_frames:],shorten=shorten)
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
		obs, _, done, info = self.env.step(action)
		self.past_observations.append(self.env.render())
		self.counter += 1
		t = self.counter/self.max_episode_steps
		if self.time:
			obs = np.concatenate([obs, np.array([t])])
		if done:
			'''frames = self.preprocess_metaworld(self.past_observations,shorten=True)
			video = th.from_numpy(frames)
			video_output = self.net(video.float())
			video_embedding = video_output['video_embedding']
			similarity_matrix = th.matmul(self.target_embedding, video_embedding.t())
			reward = similarity_matrix.detach().numpy()[0][0]'''
			reward = 0.0
		else:
			reward = 0.0  # Default reward for steps without feedback
		return obs, reward, done, info

	def reset(self):
		self.past_observations = []
		self.counter = 1
		if not self.time:
			return self.env.reset()
		return np.concatenate([self.env.reset(), np.array([0.0])])

class MetaworldDense(Env):
	def __init__(self, env_id,max_episode_steps, time=False, rank=0):
		super(MetaworldDense,self)
		door_open_goal_hidden_cls = ALL_V2_ENVIRONMENTS_GOAL_HIDDEN[env_id]
		env = door_open_goal_hidden_cls(seed=rank)
		self.max_episode_steps=max_episode_steps
		if time:
			self.env = TimeLimit(env, max_episode_steps)
		else:
			self.env=env
		self.time = time
		if not self.time:
			self.observation_space = self.env.observation_space
		else:
			self.observation_space = Box(low=-8.0, high=8.0, shape=(self.env.observation_space.shape[0]+1,), dtype=np.float32)
		self.action_space = self.env.action_space
		self.past_observations = []
		
		self.counter = 1

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
		t = self.counter/self.max_episode_steps
		if self.time:
			obs = np.concatenate([obs, np.array([t])])
		return obs, reward, done, info
		
	def reset(self):
		self.counter = 0
		if not self.time:
			return self.env.reset()
		return np.concatenate([self.env.reset(), np.array([0.0])])
		
# helper function to convert numpy arrays to tensors
def t(x):
	#print("BEFORE TRANSFORM: ",x)
	x = np.array(x) if not isinstance(x, np.ndarray) else x
	return torch.from_numpy(x).float()

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
			state.append(t(s))
			action.append(t(a))
			feedback.append(f)
			credit.append(c)

		return torch.stack(state), torch.stack(action), torch.tensor(feedback), torch.tensor(credit)

class Actor(nn.Module):
	def __init__(self, state_dim, n_actions, activation=nn.Tanh):
		super().__init__()
		self.n_actions = n_actions
		self.model = nn.Sequential(
			nn.Linear(state_dim, 256),
			activation(),
			nn.Linear(256, 256),
			activation(),
			nn.Linear(256, n_actions)
		)
		
		logstds_param = nn.Parameter(torch.full((n_actions,), 0.1))
		self.register_parameter("logstds", logstds_param)
	
	def forward(self, X,deterministic=False):
		means = self.model(X)
		if deterministic:
			# Return the means directly for deterministic action selection
			return means
		stds = torch.clamp(self.logstds.exp(), 1e-3, 50)
		
		return torch.distributions.Normal(means, stds)
	
## Critic module
class Critic(nn.Module):
	def __init__(self, state_dim, activation=nn.Tanh):
		super().__init__()
		self.model = nn.Sequential(
			nn.Linear(state_dim, 256),
			activation(),
			nn.Linear(256, 256),
			activation(),
			nn.Linear(256, 1),
		)
	
	def forward(self, X):
		output = self.model(X)
		if args.bounded==1:
			output=50 * torch.tanh(output)
		elif args.bounded==2:
			output=torch.clamp(output, -50, 50)
		return output

def clip_grad_norm_(module, max_grad_norm):
	nn.utils.clip_grad_norm_([p for g in module.param_groups for p in g["params"]], max_grad_norm)

class CreditAssignment():
	def __init__(self, dist):
		self.dist = dist
		
	def __call__(self, s_start: int, h_start: int) -> int:
		step_diff = h_start - s_start
		return self.dist.pdf(step_diff)
	

class A2CLearner():
	def __init__(self, actor, critic, queue, entropy_beta=0.001,
				 actor_lr=1e-4, critic_lr=1e-3, max_grad_norm=0.50):
		self.max_grad_norm = max_grad_norm
		self.actor = actor
		self.critic = critic
		self.entropy_beta = entropy_beta
		self.actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
		self.critic_optim = torch.optim.Adam(critic.parameters(), lr=critic_lr)
		self.queue = queue 
		self.buffer = BufferDeque(1000)
		self.sliding_window = deque()

		if args.pretrained:
			checkpoint = torch.load(args.pretrained)
			self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
			self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])


	def get_action(self,state,action_s_l,action_s_m,deterministic=False):
		dists = self.actor(t(state),deterministic)
		if deterministic:
			actions_clipped = np.clip(dists.detach().data.numpy(), action_s_l, action_s_m)
		else:
			actions = dists.sample().detach().data.numpy()
			actions_clipped = np.clip(actions, action_s_l, action_s_m)
		#print("ACTION: ",actions_clipped)
		return actions_clipped
	
	def learn(self,states, actions, fb,credits , steps):
		

		predicted_feedbacks = self.critic(states)
		predicted_feedbacks = predicted_feedbacks.squeeze(1)
		#print("PREDICTION: ",predicted_feedbacks,"FEEDBACK: ",fb)
		#print("pred_fb_shape:", predicted_feedbacks.shape)
		#print("fb_shape:", fb.shape)
		#print("credits_shape:", credits.shape)
		advantage = credits * (predicted_feedbacks - fb) ** 2
		advantage_expanded = advantage.unsqueeze(-1)

		# actor
		norm_dists = self.actor(states)
		logs_probs = norm_dists.log_prob(actions)
		entropy = norm_dists.entropy().mean()

		#print("logs_probs shape:", logs_probs.shape)
		#print("advantage shape:", advantage.shape)
		#print("entropy shape:", entropy.shape)
		
		actor_loss = (-logs_probs*advantage_expanded.detach()).mean() - entropy*self.entropy_beta
		self.actor_optim.zero_grad()
		actor_loss.backward()
		
		clip_grad_norm_(self.actor_optim, self.max_grad_norm)
		writer.add_histogram("gradients/actor",
							 torch.cat([p.grad.view(-1) for p in self.actor.parameters()]), global_step=steps)
		writer.add_histogram("parameters/actor",
							 torch.cat([p.data.view(-1) for p in self.actor.parameters()]), global_step=steps)
		self.actor_optim.step()

		# critic
		critic_loss = torch.mean(credits * (predicted_feedbacks - fb) ** 2)
		self.critic_optim.zero_grad()
		critic_loss.backward()
		clip_grad_norm_(self.critic_optim, self.max_grad_norm)
		writer.add_histogram("gradients/critic",
							 torch.cat([p.grad.view(-1) for p in self.critic.parameters()]), global_step=steps)
		writer.add_histogram("parameters/critic",
							 torch.cat([p.data.view(-1) for p in self.critic.parameters()]), global_step=steps)
		self.critic_optim.step()
		
		# reports
		writer.add_scalar("losses/log_probs", -logs_probs.mean(), global_step=steps)
		writer.add_scalar("losses/entropy", entropy, global_step=steps) 
		writer.add_scalar("losses/entropy_beta", self.entropy_beta, global_step=steps) 
		writer.add_scalar("losses/actor", actor_loss, global_step=steps)
		writer.add_scalar("losses/advantage", advantage.mean(), global_step=steps)
		writer.add_scalar("losses/critic", critic_loss, global_step=steps)



class Runner():
	def __init__(self, env,eval_env,queue,a2clearner,log_dir,VLM,evaluative,eval_freq=256):
		self.env = env
		self.state = None
		self.done = True
		self.steps = 1
		self.episode_reward = 0
		self.episode_rewards = []
		self.queue = queue
		self.buffer = BufferDeque(1000)
		self.sliding_window = deque()
		self.a2clearner = a2clearner
		self.eval_freq=eval_freq
		self.best_mean_reward = -np.inf
		self.eval_env=eval_env
		self.save_path=log_dir	
		self.VLM=VLM
		self.evaluative=evaluative

	def evaluate_model(self):
		total_rewards = []
		for _ in range(10):  # Evaluate over 10 episodes
			obs = self.eval_env.reset()
			done = False
			total_reward = 0
			while not done:
				obs_tensor = obs
				with torch.no_grad():
					self.a2clearner.actor.eval()
					action_pred = self.a2clearner.get_action(obs_tensor,self.eval_env.action_space.low.min(),self.eval_env.action_space.high.max(),True)
				obs, reward, done, _ = self.eval_env.step(action_pred)
				total_reward += reward
			total_rewards.append(total_reward)
		avg_reward = sum(total_rewards) / len(total_rewards)
		writer.add_scalar("eval_avg_reward", avg_reward, global_step=self.steps)
		self.a2clearner.actor.train()
		return avg_reward

	def feedback(self):
			fb=0
			#print("STEP_env: ",self.env.get_counter(),"STEP_runner:",self.steps)
			if self.steps % args.n_frames == 0 and self.steps!=0:
				#if self.evaluative:
				print(args.n_frames," steps reached. Please provide feedback (",self.env.get_counter()-1,"/",args.n_steps,"):")
				self.env.show_progress()
				if self.VLM:
					new_instruction = input()  # Assuming new instruction is text. Modify as needed.
					# Process new instruction
					fb = self.env.get_similarity(new_instruction,False)
					if args.bounded==1:
						fb = 50 * np.tanh(fb)
					elif args.bounded==2:
						fb=np.clip(fb, -50, 50)
				else:
					fb=int(input())
				print("Feedback: ",fb)
			writer.add_scalar("step_feedback", fb, global_step=self.steps)
			writer.flush()
			self.state_start_time = self.steps
			self.queue.put(
					dict(
						feedback = fb,
						h_time = self.steps
					))
			

	def after_set_action(self):
		batch=64
		
		if len(self.buffer) > 30:      
			# Only train every certain number of steps
			if self.steps % 16 == 0 and self.steps>0: 
				#rand_batch = np.random.randint(len(buffer), size=batch_size) 
				state, action, feedback, credit = self.buffer.random_sample(batch)
				self.a2clearner.learn(state, action, feedback,credit , self.steps) 
		
	def after_step(self):
		fb_dict = self.queue.get()
		# fb = torch.tensor(fb_dict["feedback"]).to(device)
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
			ca = CreditAssignment(norm(loc=0, scale=32/4))
			state, action, credit = [], [], []
			for win in self.sliding_window:
				if h_time-win["s_start"]>31:
					credit_for_state=0
				else:
					credit_for_state = ca(s_start=win["s_start"],h_start=h_time)
				#print("credit",credit_for_state,"s_start",win["s_start"],"h_start",h_time)
				if credit_for_state !=0:
					print("credit: ",credit_for_state, "step: ",win["s_start"])
					state.append(t(win['state']))
					action.append(t(win['action']))
					# credit.append(torch.tensor(credit_for_state, dtype=torch.float32).to(device))
					credit.append(credit_for_state) 
					self.buffer.push([
						win['state'], 
						win['action'],
						fb,
						credit_for_state
						# torch.Tensor(credit) 
					])
			#print("state",state,"\n action",action,"\n credit",credit)
			state, action, credit = torch.stack(state), torch.stack(action), torch.tensor(credit)
			feedback = torch.full(credit.size(), fb)
			self.a2clearner.learn(state, action, feedback,credit , self.steps)
				  
		if self.steps % self.eval_freq == 0:
			mean_reward = self.evaluate_model()
			if mean_reward > self.best_mean_reward:
				self.best_mean_reward = mean_reward
				torch.save({
			'actor_model_state_dict': self.a2clearner.actor.state_dict(),
			'critic_model_state_dict': self.a2clearner.critic.state_dict(),
			'actor_optim_state_dict': self.a2clearner.actor_optim.state_dict(),
			'critic_optim_state_dict': self.a2clearner.critic_optim.state_dict()},os.path.join(self.save_path, "best_model.pth"))
				print(f"New best model saved with mean reward: {mean_reward}")
		
	def reset(self):
		self.episode_reward = 0
		self.done = False
		self.state = self.env.reset()
	
	def run(self, max_steps):
		
		for i in range(max_steps):
			if self.done: self.reset()
			
			self.action=self.a2clearner.get_action(self.state,self.env.action_space.low.min(),self.env.action_space.high.max(),False)

			self.after_set_action()


			next_state, reward, self.done, info = self.env.step(self.action)
			self.feedback()
			self.after_step()

			self.state = next_state
			self.steps += 1
			self.episode_reward += reward
			
			if self.done:
				self.episode_rewards.append(self.episode_reward)
				if len(self.episode_rewards) % 10 == 0:
					print("episode:", len(self.episode_rewards), ", episode reward:", self.episode_reward)
				writer.add_scalar("episode_reward", self.episode_reward, global_step=self.steps)
				break
					

def make_env(env_type, env_id,max_episode_steps, text_string,time,rank, seed=0):
	"""
	Utility function for multiprocessed env.

	:param env_id: (str) the environment ID
	:param num_env: (int) the number of environments you wish to have in subprocesses
	:param seed: (int) the inital seed for RNG
	:param rank: (int) index of the subprocess
	"""
	# env = KitchenMicrowaveHingeSlideV0()
	if env_type=="interactive":
		env= MetaworldInteractive(env_id=env_id,max_episode_steps=max_episode_steps, text_string=text_string, time=time, rank=rank, human=True)
	else:
		env = MetaworldDense(env_id=env_id,max_episode_steps=max_episode_steps, time=time, rank=rank)
	#env = Monitor(env, os.path.join(log_dir, str(rank)))
	# env.seed(seed + rank)
	return env

def main():
	global args
	global log_dir
	global writer
	args = get_args()
	log_dir = f"metaworld/{args.env_id}_{args.env_type}{args.dir_add}"
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	writer = SummaryWriter(log_dir)
	timed=bool(args.timed)
	VLM=bool(args.VLM)
	evaluative=bool(args.evaluative)

	env = make_env(args.env_type, args.env_id,args.n_steps,args.text_string,timed, 0)
	eval_env=make_env("dense_original", args.env_id,args.n_steps,None,timed, 0)
	state_dim = env.observation_space.shape[0]
	n_actions = env.action_space.shape[0]
	actor = Actor(state_dim, n_actions,activation=Mish)
	critic = Critic(state_dim,activation=Mish)
	if args.pretrained:
		checkpoint = torch.load(args.pretrained)
		actor.load_state_dict(checkpoint['actor_model_state_dict'])
		critic.load_state_dict(checkpoint['critic_model_state_dict'])
		actor.train()
		critic.train()
	
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

	
	Feedback_queue = Queue()
	learner = A2CLearner(actor, critic,Feedback_queue)
	print("VLM: ",VLM)
	runner = Runner(env,eval_env,Feedback_queue,learner,log_dir,VLM,evaluative,args.n_steps)
	
	while runner.steps<args.total_time_steps:
		runner.run(args.n_steps)
		torch.save({
				'actor_model_state_dict': learner.actor.state_dict(),
				'critic_model_state_dict': learner.critic.state_dict(),
				'actor_optim_state_dict': learner.actor_optim.state_dict(),
				'critic_optim_state_dict': learner.critic_optim.state_dict()},os.path.join(log_dir, "trained.pth"))

	

if __name__ == "__main__":
	main() 
