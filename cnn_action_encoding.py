import argparse
import numpy as np
import os
import random

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
from ray.rllib.models.torch.torch_distributions import TorchCategorical, TorchDistribution, TorchDiagGaussian 

from typing import Dict

from ray.rllib import Policy
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode
from ray.rllib.utils.typing import PolicyID
from ray.tune import with_resources



OBSERVATION_SHAPE = (4, 64, 64)  # just a random observation shape
# OBSERVATION_SHAPE = (64, 64, 4)  # just a random observation shape
# OBSERVATION_SHAPE = (42, 42, 2)  # just a random observation shape
# tf1, tf, tfv = try_import_tf()
torch, nn = try_import_torch()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=float, default=5000, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=float, default=5000000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward", type=float, default= 9000, help="Reward at which we stop training."
)
parser.add_argument(
    "--no-tune",
    action="store_true",
    help="Run without Tune using a manual train loop instead. In this case,"
    "use PPO without grid search and no TensorBoard.",
)
parser.add_argument(
    "--local-mode",
    action="store_true",
    help="Init Ray in local mode for easier debugging.",
)


torch, nn = try_import_torch()

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
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
    
    def _forward(self, input_dict, **kwargs):        
        return {ENCODER_OUT: (self.net(input_dict["obs"]))}


class CNNActorHeadConfig(ModelConfig):

    def build(self, framework: str = "torch") -> "Model":

        return CNNActorHead(self)

class CNNActorHead(TorchModel):

    def __init__(self, config: CNNActorHeadConfig) -> None:
        super().__init__(config)

        self.cnn_transpose_net = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(128)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 2, kernel_size=4, stride=2, padding=1),
            # nn.BatchNorm2d(64)
            nn.ReLU(), 
            nn.Flatten(1, -1)
        )
            
    def _forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        print("inside cnn actor", inputs.shape)
        out = self.cnn_transpose_net(inputs)

        # out = out.permute(0, 2, 3, 1)
        # Add 0.5 to center (always non-activated, non-normalized) outputs more
        # around 0.0.
        print("inside cnn actor out", out.shape)
        return out 

class CNNCriticHeadConfig(ModelConfig):

    def build(self, framework: str = "torch") -> "Model":

        return CNNCriticHead(self)

class CNNCriticHead(TorchModel):

    def __init__(self, config: CNNCriticHeadConfig) -> None:
        super().__init__(config)

        self.cnn_transpose_net = nn.Sequential(
            nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.ReLU(), 
            nn.AvgPool2d(kernel_size=64),
            nn.Flatten(1, -1)
        )
            
    def _forward(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        print("inside cnn critic", inputs.shape)
        out = self.cnn_transpose_net(inputs)
        print("inside cnn critic out", out.shape)
        # Add 0.5 to center (always non-activated, non-normalized) outputs more
        # around 0.0.
        return out

class CustomTorchPPORLModule(PPOTorchRLModule):
    # def __init__(self, config: RLModuleConfig):
    #     super().__init__(config)
    
    def setup(self):

        custom_encoder_config = CustomEncoderConfig(num_channels=self.config.model_config_dict['num_channels'])
        actor_critic_encoder_config = ActorCriticEncoderConfig(base_encoder_config=custom_encoder_config, shared=self.config.model_config_dict['shared'])
        self.encoder = actor_critic_encoder_config.build(framework="torch")


        pi_config = CNNActorHeadConfig()
        vf_config = CNNCriticHeadConfig()

        self.pi = pi_config.build(framework="torch")
        self.vf = vf_config.build(framework="torch")

        #self.action_dist_cls = TorchDiagGaussian TorchDeterministic
        self.action_dist_cls =  TorchDiagGaussian

class CustomMetricCallbacks(DefaultCallbacks):

    @staticmethod
    def get_info(base_env, episode):
        """Return the info dict for the given base_env and episode"""
        # different treatment for MultiAgentEnv where we need to get the info dict from a specific UE
        if hasattr(base_env, 'envs'):
            # get the info dict for the first UE (it's the same for all)
            ue_id = base_env.envs[0].ue_list[0].id
            info = episode.last_info_for(ue_id)
        else:
            info = episode.last_info_for()
        return info

    def on_episode_step(self, *, worker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        info = self.get_info(base_env, episode)
        # add all custom scalar metrics in the info dict
        if info is not None and 'scalar_metrics' in info:
            for metric_name, metric_value in info['scalar_metrics'].items():
                
                episode.custom_metrics[metric_name] = metric_value

                # increment (or init) the sum over all time steps inside the episode
                eps_metric_name = f'eps_{metric_name}'
                if eps_metric_name in episode.user_data:
                    episode.user_data[eps_metric_name] = metric_value
                else:
                    episode.user_data[eps_metric_name] = metric_value

    @staticmethod
    def on_episode_end(*, worker, base_env: BaseEnv,
                       policies: Dict[PolicyID, Policy],
                       episode: MultiAgentEpisode, env_index: int, **kwargs):
        # log the sum of scalar metrics over an episode as metric
        for key, value in episode.user_data.items():
            episode.custom_metrics[key] = value



class FluidEnv(gym.Env):

    def __init__(self, config: EnvContext):
    

        self.time = config["time"]
        self.count = 0

        self.observation_space = gym.spaces.Box(low=-50.0, high=50.0, shape=OBSERVATION_SHAPE, dtype=np.float16)
        self.action_space = gym.spaces.Box(low=-1.0, high=0.0, shape=(4096,), dtype = np.float16)
        
        self.reset()
        self.seed()

        self.global_steps = 0


    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, *, seed = None, options = None):

        # Initialize the grid and current time
        # flag = [0]
        # client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # client_socket.connect(server_address)
        # data = pickle.dumps(flag)
        # client_socket.sendall(data)
        # received_data = client_socket.recv(52428800)
        # processed_result = pickle.loads(received_data)
        #print(f"data received inside reset = {processed_result}", flush = True)
        processed_result = np.random.random(OBSERVATION_SHAPE)
        self.count = 0
        return processed_result


    def step(self, action):

        epsilon_1 = 1
        epsilon_2 = 35
        C1 = 100
        C2 = 100
        BETA = 0.5

        # Get the control variable (activity coefficient) from the action
        activity_coeff = action.reshape(64, 64)
        print("activity shape", np.shape(activity_coeff))
        data_to_send = activity_coeff.tolist()
        
        self.global_steps += 1
        print(f"Global steps: {self.global_steps:5d}; Action: {np.mean(data_to_send):2.4f}")

        ###################### TIME STEPPING #############################
        # client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # client_socket.connect(server_address)
        # data = pickle.dumps(data_to_send)
        # # print(f"sending data to server at {self.count}", flush = True)
        # client_socket.sendall(data)
        # received_data = client_socket.recv(52428800)
        # processed_result = pickle.loads(received_data)
        # #################################################################
        # self.grid = np.array(processed_result)
        self.grid = np.random.random(OBSERVATION_SHAPE)
        print("observation shape", np.shape(self.grid))
        ########################## CALCULATE REWARD ######################
        arr = self.grid[0, :, :]
        middle_start_col = ((arr.shape[0] - int((arr.shape[0]/50)*10)) // 2 ) + 1
        ux_mid = arr[middle_start_col:middle_start_col + int((arr.shape[0]/50)*10), :]
        mean_ux = np.mean(ux_mid)
        reward = C1*np.power(BETA, C2*(mean_ux - 0.4)**2)
        ###################################################################
        # reward = np.random.rand()

        if ( reward < epsilon_1 and self.count > self.time):         # too bad
            done = True

        elif ( reward > epsilon_2 and self.count > self.time): # good enough 
            done = True
        else:             
            self.count = self.count + 1     # continue learning 
            done = False



        info_dict = {
            "scalar_metrics": {"mean_ux": mean_ux}
        }
        return (
            self.grid,
            reward,
            done,
            info_dict,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    print(f"\nRunning with following CLI options: {args}\n")
    
  
    ray.init(
        local_mode=args.local_mode
    )
    
    config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .experimental(_disable_preprocessor_api=True)
        .rl_module(
            rl_module_spec=SingleAgentRLModuleSpec(
                module_class=CustomTorchPPORLModule,
                model_config_dict={"num_channels": [16,32,256], "shared": False}
            )
        )
        .environment(FluidEnv, env_config={
            "time": 50}
        )
        .framework(args.framework)
        .rl_module(_enable_rl_module_api=True)
        .training(_enable_learner_api=True)
        .rollouts(num_rollout_workers=1)
        .resources(
            num_learner_workers=1,
            num_gpus_per_learner_worker=1,
            num_cpus_per_worker=4, 
            num_gpus_per_worker=0.5
        )        
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .callbacks(CustomMetricCallbacks)        
    )


    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }


    # # Define the address and port for communication
    server_address = ('localhost', 12345)

    if args.no_tune:
        # manual training with train loop using PPO and fixed learning rate
        if args.run != "PPO":
            raise ValueError("Only support --run PPO with --no-tune.")
        print("Running manual train loop without Ray Tune.")
        # use fixed learning rate instead of grid search (needs tune)
        config.lr = 1e-3
        algo = config.build()
        # run manual training loop and print results after each iteration
        for i in range(args.stop_iters):
            result = algo.train()

            if (
                result["timesteps_total"] >= args.stop_timesteps
                or result["episode_reward_mean"] >= args.stop_reward
            ):
                break
        algo.stop()
    
    else:      
        param_space = config.to_dict()
        vf_clip_param = 20
        lr = 10**(-4)
        param_space.update({
            "sgd_minibatch_size": 128,
            "train_batch_size": 1024,
            "vf_clip_param" : vf_clip_param,
            "lr" : lr,
        })


        print("Training automatically with Ray Tune")
        tuner = tune.Tuner(
            args.run,
            param_space=param_space,
            run_config=air.RunConfig(
                stop=stop, 
                checkpoint_config=air.CheckpointConfig(checkpoint_frequency=15, checkpoint_at_end=True),
                verbose=3, 
                storage_path = f"/scratch0/saptorshighosh/ray_results",
                # name = f"test_sigmoid_lr_{lr}"
            ),
        )
        results = tuner.fit()
    
        if args.as_test:
            print("Testing the agent...")
            print("Checking if learning goals were achieved")
            check_learning_achieved(results, args.stop_reward)
        
    ray.shutdown()


