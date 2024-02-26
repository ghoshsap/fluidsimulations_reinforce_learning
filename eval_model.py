import argparse
import gymnasium as gym
from gymnasium.spaces import Box

import numpy as np
import os
import random

import socket
import pickle

import ray
from ray import air, tune
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
from ray.rllib.algorithms.algorithm import Algorithm

from typing import Dict

from ray.rllib import Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.typing import PolicyID
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import with_resources
import torch
from scipy.ndimage import gaussian_filter

obs_shape = (2, 42, 42)
nx, ny = 26, 26
x = np.zeros(obs_shape).astype(np.float16)
x_bytes = pickle.dumps(x) 

directory_path = f"/home/saptorshi/Disk/ray_model_test/time_300_checkpoint_244_director"

if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    print(f"Directory '{directory_path}' created.")
else:
    print(f"Directory '{directory_path}' already exists.")


class ActiveNemEnv():

    def __init__(self):

        self.obs = None
        self.global_steps = 0
        self.time = 51

    def reset(self, *, seed = None, options = None):

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port = 1234
        s.connect(('127.0.0.1', port))

        f = np.zeros([nx, ny]).astype(np.float16)
        data_to_send = pickle.dumps(f) 
        while data_to_send:
            sent = s.send(data_to_send)
            data_to_send = data_to_send[sent:]

        # self.obs = np.zeros(obs_shape, dtype=np.float16)
        self.obs = np.random.random(obs_shape).astype(np.float16)
        self.count = 0
        return self.obs

        

    def step(self, action):

        epsilon_1 = 1
        epsilon_2 = 35
        C1 = 100
        C2 = 100
        BETA = 0.5

        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port = 1234
        s.connect(('127.0.0.1', port))

        # Get the control variable (activity coefficient) from the action
        action = action.astype(np.float16)
        activity_coeff = action.reshape(nx, ny)
        data_to_send = pickle.dumps(activity_coeff)

        self.global_steps += 1
        print(f"Global steps: {self.global_steps:5d}; Action: {np.mean(activity_coeff):2.4f}")

        ###################### TIME STEPPING ##############################################
        while data_to_send:
            sent = s.send(data_to_send)
            data_to_send = data_to_send[sent:]


        received_data = b""
        while len(received_data) < len(x_bytes):
            chunk = s.recv(len(x_bytes) - len(received_data))
            if not chunk:
                raise RuntimeError("socket connection broken")
            received_data += chunk

        raw_data = pickle.loads(received_data).reshape(obs_shape)
        ####################################################################################
        # print(f"average at recv at {self.global_steps:5d}", np.sum(self.grid))
        norm_1 = (raw_data[0,:,:] - np.min(raw_data[0,:,:]))/(np.max(raw_data[0,:,:]) - np.min(raw_data[0,:,:]))
        norm_2 = (raw_data[1,:,:] - np.min(raw_data[1,:,:]))/(np.max(raw_data[1,:,:]) - np.min(raw_data[1,:,:]))

        self.grid = np.stack((norm_1, norm_2), axis = 0)

        ########################## CALCULATE REWARD ######################
        ux = raw_data[0, :, :]
        uy = raw_data[1, :, :]
    
        middle_start_col = ( (ux.shape[0] - int((ux.shape[0]/50)*10)) // 2 ) + 1
        ux_mid = ux[middle_start_col:middle_start_col + int((ux.shape[0]/50)*10), :]
        uy_mid = uy[middle_start_col:middle_start_col + int((ux.shape[0]/50)*10), :]

        x =  (np.mean(ux_mid) - 0.4)**2 

        reward = C1*np.power(BETA, C2*x)

        mean_ux = np.mean(ux_mid)
        ###################################################################
        
        if ( reward < epsilon_1 and self.count > self.time):         # too bad
            done = True

        elif ( reward > epsilon_2 and self.count > self.time): # good enough 
            done = True
        else:             
            self.count = self.count + 1     # continue learning 
            done = False

        # print(self.grid)

        info_dict = {
            "scalar_metrics": {"mean_ux": mean_ux}
        }
        return (
            self.grid,
            reward,
            done,
            info_dict,
        )


env = ActiveNemEnv()

ray.init()

server_address = ('localhost', 1234)
checkpoint = f"/home/saptorshi/Disk/ray_results/baseline_test_new_feb10/PPO_FluidEnv_37691_00000_0_2024-02-10_02-08-39/checkpoint_000244"
algo = Algorithm.from_checkpoint(checkpoint)
model_ = algo.get_policy().model

model = model_.encoder.actor_encoder.net
layers_mu = model_.pi.actor_mu



reward_episode = []

for k in range(1):

    u_old = env.reset()
    state_list = []
    action_list = []
    reward_list = []

    for _ in range(300):

        outputs = []
        x = torch.tensor(u_old).to(dtype=torch.float32).cuda()

        print(x.shape)

        with torch.no_grad():
            for i,j in enumerate(model):
                x = model[i](x)
                outputs.append(x)

        arr = outputs[8].cpu().numpy()
        x = torch.tensor(arr.flatten()).to(dtype=torch.float32).cuda()

        with torch.no_grad():
            x = layers_mu(x)

        activity_mean = x.cpu().numpy()

        action = gaussian_filter(activity_mean.reshape(26, 26), sigma = 0.2)

        state_list.append(u_old)
        action_list.append(action)

        u_new, reward, done, info = env.step(action)
        print(f"u_new shape = ", u_new.shape)
        print(f"reward = ", reward)
        u_old = u_new
        reward_list.append(reward)
        
    total_reward = np.sum(reward_list)
    reward_episode.append(total_reward)

    print(f"at episode {k} the reward obtained was {total_reward}", flush = True)

    np.savez(f"{directory_path}/trajectory_{k}.npz", state_list = state_list, action_list = action_list, reward_list = reward_list, total_reward = np.sum(reward_list))
np.savez(f"{directory_path}/reward_episode.npz", reward_episode = reward_episode)

ray.shutdown()
