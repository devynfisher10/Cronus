
import numpy as np
from rlgym.envs import Match
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy


from rlgym.utils.obs_builders import AdvancedObs
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogRewardCallback


from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward, VelocityReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import LiuDistancePlayerToBallReward
from rewards import TouchVelChange,JumpTouchReward,DoubleTapReward, AirDribbleReward, AerialReward



# using Leaky Relu activation function
from torch.nn import LeakyReLU

from state import CronusStateSetter
from terminal import CronusTerminalCondition
from sprinter_parser import LookupAction

if __name__ == '__main__':  # Required for multiprocessing
    frame_skip = 8          # Number of ticks to repeat an action
    half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5

    # initial parameters, will update over the course of learning to increase batch size and comment iterations
    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  
    # start steps at 500_000 instead of 1 mil for early learning speed, move to 1 mil later
    target_steps = 1_000_000
    agents_per_match = 2
    # 7 is number of instances that maximizes fps on my system
    num_instances=7
    steps = target_steps // (num_instances * agents_per_match) #making sure the experience counts line up properly
    batch_size = 100_000
    loading_model=True #check that loading model instead of starting from scratch
    model_to_load = "exit_save.zip" #exit_save.zip rl_model_281069544_steps
    print(f"fps={fps}, gamma={gamma})")



    def get_match():  # Need to use a function so that each instance can call it and produce their own objects
        goal_weight = 10 # 
        demo_weight = 3 # 
        boost_weight = .05 # 

        # defining initial custom reward weights, will update over time for curriculum learning and comment iterations
        # v0.1
        event_weight = 1
        touch_vel_weight = 15 # from 18 at 450 mil
        vel_ball_weight = 2
        vel_weight = .00025 # from .00005 at 450 mil
        jump_touch_weight = 40 # from 9 at 450 mil
        double_tap_weight = 1 # 
        air_dribble_weight = .1 # 
        aerial_weight = .05 # from .008 at 450 mil
        dist_to_ball_weight = .001
 

        return Match(
            team_size=1,  # 1v1 bot only
            tick_skip=frame_skip,
            reward_function=SB3CombinedLogReward(
                (
                 EventReward(goal=goal_weight, concede=-goal_weight, demo=demo_weight, boost_pickup=boost_weight),  
                 TouchVelChange(),
                 VelocityBallToGoalReward(),
                 VelocityReward(),
                 JumpTouchReward(),
                 DoubleTapReward(),
                 AirDribbleReward(),
                 AerialReward(),
                 LiuDistancePlayerToBallReward(),
                 ),
                (event_weight, touch_vel_weight, vel_ball_weight, vel_weight, jump_touch_weight, double_tap_weight, air_dribble_weight, aerial_weight, dist_to_ball_weight),
                "logs"
            ),
            spawn_opponents=True,
            terminal_conditions=[CronusTerminalCondition()],
            obs_builder=AdvancedObs(),  
            state_setter=CronusStateSetter(),  
            action_parser=LookupAction(),
        )

    # creating env and force paging 
    env = SB3MultipleInstanceEnv(get_match, num_instances, wait_time=80, force_paging=True)            
    env = VecCheckNan(env)                                # Optional
    env = VecMonitor(env)                                 # Recommended, logs mean reward and ep_len to Tensorboard
    env = VecNormalize(env, norm_obs=False, gamma=gamma)  # Highly recommended, normalizes rewards

    # we try loading existing model and if does not exist then we create new model
    try:
        assert loading_model
        model=PPO.load(
            f"models/{model_to_load}",
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
            net_arch=[512, dict(pi=[256,256], vf=[256, 256, 256])]
            
         )
        model = PPO(
            MlpPolicy,
            env,
            n_epochs=10,                 # had to drop epochs for fps gain, v0.1=30, v0.2=10
            policy_kwargs=policy_kwargs,
            learning_rate=5e-5,          
            ent_coef=0.01,               # From PPO Atari
            vf_coef=1.,                  # From PPO Atari
            gamma=gamma,                 # Gamma as calculated using half-life
            clip_range=0.2,
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
    reward_list=['event', 'touch_vel','vel_ball','vel','jump_touch','double_tap', 'air_dribble', 'aerial', 'dist_to_ball']

    save_callback = CheckpointCallback(round(5_000_000 / env.num_envs), save_path="models", name_prefix="rl_model_v2")
    reward_callback = SB3CombinedLogRewardCallback(reward_list, 'logs')
    callbacks = CallbackList([save_callback, reward_callback])
    try:
        while True:
            model.learn(100_000_000, callback=callbacks, reset_num_timesteps=False, tb_log_name="PPO_2_1")

            model.save("models/exit_save")
            model.save("mmr_models/" + str(model.num_timesteps))
    
    except KeyboardInterrupt:
        model.save("models/exit_save")
        model.save("mmr_models/" + str(model.num_timesteps))
        print("Exiting training")


