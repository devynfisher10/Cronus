
import numpy as np
from rlgym.envs import Match
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy
from rlgym_tools.extra_obs.advanced_stacker import AdvancedStacker
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

# using Leaky Relu activation function and defining network shape
from torch.nn import LeakyReLU

from rewards import CronusRewards
from state import CronusStateSetter
from terminal import CronusTerminalCondition
from sprinter_parser import LookupAction

if __name__ == '__main__':  # Required for multiprocessing
    frame_skip = 8          # Number of ticks to repeat an action
    half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5

    # initial parameters, will update over the course of learning to increase batch size and comment iterations
    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  
    target_steps = 1_000_000
    agents_per_match = 2
    # number of instances that maximizes fps on my system
    num_instances=7
    steps = target_steps // (num_instances * agents_per_match) #making sure the experience counts line up properly
    # v0 batch size = 10,000
    batch_size = 10_000
    print(f"fps={fps}, gamma={gamma})")


    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        return Match(
            team_size=1,  # 1v1 bot only
            tick_skip=frame_skip,
            reward_function=CronusRewards(),
            spawn_opponents=True,
            terminal_conditions=[CronusTerminalCondition()],
            obs_builder=AdvancedStacker(4),  
            state_setter=CronusStateSetter(),  
            action_parser=LookupAction()
        )

    # creating env and force paging 
    env = SB3MultipleInstanceEnv(get_match, num_instances, wait_time=60, force_paging=True)            
    env = VecCheckNan(env)                                # Optional
    env = VecMonitor(env)                                 # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Highly recommended, normalizes rewards

    # we try loading existing model and if does not exist then we create new model
    try:
        model=PPO.load(
            "models/exit_save.zip",
            env,
            device="cpu",
            custom_objects={"n_envs": env.num_envs}, #automatically adjusts to users changing instance count, may encounter shaping error otherwise
            )
        print("loaded model")
        print(f"Current timesteps: {model.num_timesteps}")
    except:
        print("creating model")
        
        policy_kwargs = dict(
            activation_fn=LeakyReLU,
            net_arch=[512, 512, dict(pi=[512,256,256], vf=[512, 256, 256])]
            
         )
        model = PPO(
            MlpPolicy,
            env,
            n_epochs=25,                 # PPO calls for multiple epochs
            policy_kwargs=policy_kwargs,
            learning_rate=1e-4,          # Around this is fairly common for PPO
            ent_coef=0.01,               # From PPO Atari
            vf_coef=1.,                  # From PPO Atari
            gamma=gamma,                 # Gamma as calculated using half-life
            verbose=3,                   # Print out all the info as we're going
            batch_size=batch_size,             
            n_steps=steps,                # Number of steps to perform before optimizing network
            tensorboard_log="logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
            device="cpu",           # Uses GPU if available
        )
    print("running")

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    # This saves to specified folder with a specified name
    callback = CheckpointCallback(round(100_000 / env.num_envs), save_path="models", name_prefix="rl_model")

    try:
        while True:
            model.learn(25_000_000, callback=callback, reset_num_timesteps=False)
            model.save("models/exit_save")
            model.save("mmr_models/" + str(model.num_timesteps))
    
    except KeyboardInterrupt:
        model.save("models/exit_save")
        model.save("mmr_models/" + str(model.num_timesteps))
        print("Exiting training")