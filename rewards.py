
from rlgym.utils.reward_functions import RewardFunction, CombinedReward
from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward, AlignBallGoal, VelocityReward
from rlgym.utils.common_values import BALL_MAX_SPEED, CAR_MAX_SPEED, BLUE_GOAL_BACK, \
    BLUE_GOAL_CENTER, ORANGE_GOAL_BACK, ORANGE_GOAL_CENTER, BLUE_TEAM, ORANGE_TEAM
from rlgym.utils.math import cosine_similarity


class TouchVelChange(RewardFunction):
    """Reward for changing velocity on ball, from KaiyoBot"""
    def __init__(self):
        self.last_vel = np.zeros(3)

    def reset(self, initial_state: GameState):
        self.last_vel = np.zeros(3)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if player.ball_touched:
            vel_difference = abs(np.linalg.norm(self.last_vel - state.ball.linear_velocity))
            reward = 1.0*vel_difference / BALL_MAX_SPEED

        self.last_vel = state.ball.linear_velocity

        return reward


class MaintainVel(RewardFunction):
    """Reward for maintaining car's velocity"""
    def __init__(self):
        self.last_vel = np.zeros(3)

    def reset(self, initial_state: GameState):
        self.last_vel = np.zeros(3)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:

        vel_difference = abs(np.linalg.norm(self.last_vel - player.car_data.linear_velocity))
        # reward is negative, because value is larger if big change in velocity
        reward = -1.0*vel_difference / CAR_MAX_SPEED

        self.last_vel = player.car_data.linear_velocity

        return reward


class CronusRewards(RewardFunction):
    def __init__(self):
        super().__init__()
        # defining initial event reward weights, will update over time for curriculum learning and comment iterations
        self.goal_weight = 10
        self.demo_weight = 4
        self.boost_weight = 2
        self.save_weight = 2
        self.shot_weight=1

        # defining initial custom reward weights, will update over time for curriculum learning and comment iterations
        self.event_weight = 1
        self.touch_vel_weight = .5
        self.align_weight = .25
        self.vel_ball_weight = .05
        self.vel_weight = .025
        self.maintain_vel_weight = .1

        self.reward = CombinedReward(
            (
             EventReward(goal=self.goal_weight, concede=-self.goal_weight, demo=self.demo_weight, boost_pickup=self.boost_weight, save=self.save_weight, shot=self.shot_weight),  # 1.0
             TouchVelChange(),
             AlignBallGoal(),
             VelocityBallToGoalReward(),
             VelocityReward(),
             MaintainVel()

             ),
            (self.event_weight, self.touch_vel_weight, self.align_weight, self.vel_ball_weight, self.vel_weight, self.maintain_vel_weight))

    def reset(self, initial_state: GameState) -> None:
        self.reward.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.reward.get_reward(player, state, previous_action)
