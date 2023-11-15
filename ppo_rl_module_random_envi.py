
import socket
import pickle

import gymnasium as gym
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.distributions as distributions
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import ModelConfig, MLPHeadConfig, ActorCriticEncoderConfig
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.models.base import Encoder, ENCODER_OUT
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec, RLModuleConfig
from ray.rllib.examples.env.random_env import RandomEnv
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule

import ray
from ray import air, tune
from ray.rllib.env.env_context import EnvContext
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from typing import Dict

from ray.rllib import Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.typing import PolicyID
from ray.tune import with_resources

OBSERVATION_SHAPE = (2, 42, 42)  # just a random observation shape

class CustomEncoderConfig(ModelConfig):
    
    def __init__(self, num_channels):
        self.num_channels = num_channels

    def build(self, framework):
        # Your custom encoder
        return CustomEncoder(self)


class CustomEncoder(TorchModel, Encoder):
    def __init__(self, config):
        super().__init__(config)

        self.net = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_channels=OBSERVATION_SHAPE[0], out_channels=config.num_channels[0], kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(in_channels=config.num_channels[0], out_channels=config.num_channels[1], kernel_size=(4, 4), stride=(2, 2)),
            nn.ReLU(),
            nn.ZeroPad2d((5, 5, 5, 5)),
            nn.Conv2d(in_channels=config.num_channels[1], out_channels=config.num_channels[2], kernel_size=(11, 11), stride=(1, 1)),
            nn.ReLU(),
            nn.Flatten(start_dim = 1, end_dim = -1)
        )  
    
    def _forward(self, input_dict, **kwargs):        
        return {ENCODER_OUT: (self.net(input_dict["obs"]))}


class CustomTorchPPORLModule(PPOTorchRLModule):
    # def __init__(self, config: RLModuleConfig):
    #     super().__init__(config)
    
    def setup(self):
        # Usage
        custom_encoder_config = CustomEncoderConfig(num_channels=self.config.model_config_dict['num_channels'])
        actor_critic_encoder_config = ActorCriticEncoderConfig(base_encoder_config=custom_encoder_config, shared=self.config.model_config_dict['shared'])
        self.encoder = actor_critic_encoder_config.build(framework="torch")

        output_dims = [30976]

        pi_config = MLPHeadConfig(
            input_dims=output_dims,
            hidden_layer_dims=[], 
            output_layer_dim=2,
            output_layer_activation="tanh",
        )

        vf_config = MLPHeadConfig(
            input_dims=output_dims, output_layer_dim=1
        )

        self.pi = pi_config.build(framework="torch")
        self.vf = vf_config.build(framework="torch")

        self.action_dist_cls = TorchCategorical


config = (
    PPOConfig()
    .rl_module(
        rl_module_spec=SingleAgentRLModuleSpec(
            module_class=CustomTorchPPORLModule,
            model_config_dict={"num_channels": [16,32,256], "shared": False}
        )
    )
    .environment(
        RandomEnv,
        env_config={
            "action_space": gym.spaces.Discrete(2),
            # Test a simple Image observation space.
            "observation_space": gym.spaces.Box(
                0.0,
                1.0,
                shape=OBSERVATION_SHAPE,
                dtype=np.float32,
            ),
        },
    )
    .training(
        train_batch_size=32, 
        sgd_minibatch_size=16, 
        num_sgd_iter=10,
    )
)

algo = config.build()

model_ = algo.get_policy().model
print(model_)
config.build().train()