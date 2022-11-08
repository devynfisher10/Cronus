
from rlgym.utils.reward_functions import RewardFunction, CombinedReward
from rlgym.utils.gamestates import GameState, PlayerData
import numpy as np
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions.common_rewards.misc_rewards import EventReward, AlignBallGoal, VelocityReward
from rlgym.utils.common_values import BALL_MAX_SPEED, CAR_MAX_SPEED, BLUE_GOAL_BACK, \
    BLUE_GOAL_CENTER, ORANGE_GOAL_BACK, ORANGE_GOAL_CENTER, BLUE_TEAM, ORANGE_TEAM, BACK_WALL_Y, BALL_RADIUS
#from rlgym_tools.extra_rewards import JumpTouchReward
from rlgym_tools.sb3_utils.sb3_log_reward import SB3CombinedLogReward


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

class BadTurtle(RewardFunction):
    """Negative reward for being on the ground on a part of the car that is not the wheels"""
    def __init__(self):
        pass

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:

        car_height = player.car_data.position[2] 
        # reward is negative, because turtling is not generally useful
        # car height between on wheels resting value and slightly above turtling resting value and z portion of vector normal to car indicates car is upside down
        if (car_height> 20.0) and (car_height<=50.0) and (not player.on_ground) and (player.car_data.up()[2] < 0):
            reward = -1.0
        else:
            reward=0

        return reward


class JumpTouchReward(RewardFunction):
    def __init__(self, min_height=92.75):
        self.min_height = min_height
        self.max_height = 2044-92.75
        self.range = self.max_height - self.min_height

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched and not player.on_ground and state.ball.position[2] >= self.min_height:
            return (state.ball.position[2] - self.min_height) / self.range

        return 0


class AerialReward(RewardFunction):
    """Rewards every step car is in air and away from walls. Emergency reward to encourage getting unstuck from ground. Will remove / severely tone down once is getting jump touches."""
    def __init__(self, min_height=25):
        self.min_height = min_height
        self.max_height = 2044-92.75
        self.range = self.max_height - self.min_height
        self.prev_has_flip = True

    def reset(self, initial_state: GameState):
        self.prev_has_flip = True

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        # reward if car off ground above min height and away from any walls
        if not player.on_ground and player.car_data.position[2] >= self.min_height and abs(player.car_data.position[1]) <= 4000 and abs(player.car_data.position[0]) <= 3000 and self.prev_has_flip and not player.has_flip:
            self.prev_has_flip = player.has_flip
            return .5

        self.prev_has_flip = player.has_flip
        return 0


class DoubleTapReward(RewardFunction):
    """Class to reward agent for behavior related to double taps. Agent gets reward if making air touch after ball hits backboard before the ball hits the ground"""
    def __init__(self):
        # off_backboard and air_touch vars initialized to False. Set to true when each event occurrs, set back to false when ball touches ground
        self.off_backboard = False
        self.first_air_touch = False
        self.second_air_touch = False
        self.min_height = BALL_RADIUS + 5
        self.min_backboard_height = 500 # top of goal is 642.775, changed from 250
        self.min_car_dist_from_backboard = BALL_RADIUS*6 # from 2 at 550 mil
        self.num_steps = 0
        self.prev_ball_vel = np.zeros(3)

    def reset(self, initial_state: GameState):
        self.off_backboard = False
        self.first_air_touch = False
        self.second_air_touch = False
        self.num_steps = 0
        self.prev_ball_vel = np.zeros(3)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        # reward if shot in air before hitting ground after backboard touch
        reward = 0

        # need to account for attacking backboard based on team
        if player.team_num == BLUE_TEAM:
            ball_position = state.ball.position
        else:
            ball_position = state.inverted_ball.position

        # need to account for order, only reward for backboard -> air touch not other way around otherwise that just rewards hitting ball into backboard

        # if ball hits backboard, and previous ball velocity was mostly towards the backboard, set off_backboard value. Requires height to be in air
        if (ball_position[2] >= self.min_backboard_height) and (abs(self.prev_ball_vel[1]) > abs(self.prev_ball_vel[0]) + (abs(self.prev_ball_vel[2]))) and (ball_position[1] >= BACK_WALL_Y - BALL_RADIUS - 10):
            self.off_backboard=True

        # if make air touch, set first_air_touch value. Only set this value if has not yet touched the backboard 
        if (not self.off_backboard) and (player.ball_touched) and (ball_position[2] >= self.min_height):
            # adding checks to make sure car is a min distance from wall when making touches to prevent dribbling on wall
            if (player.team_num == ORANGE_TEAM) and (abs(-1.0*BACK_WALL_Y - player.car_data.position[1]) >= self.min_car_dist_from_backboard):
                self.first_air_touch=True
            elif (player.team_num == BLUE_TEAM) and (abs(1.0*BACK_WALL_Y - player.car_data.position[1]) >= self.min_car_dist_from_backboard):
                self.first_air_touch=True


        # if make air touch, set second_air_touch value. Only set this value if has already touched the backboard 
        if self.off_backboard and player.ball_touched and ball_position[2] >= self.min_height:
            # adding checks to make sure car is a min distance from wall when making touches to prevent dribbling on wall
            if (player.team_num == ORANGE_TEAM) and (abs(-1.0*BACK_WALL_Y - player.car_data.position[1]) >= self.min_car_dist_from_backboard):
                self.second_air_touch=True
            elif (player.team_num == BLUE_TEAM) and (abs(1.0*BACK_WALL_Y - player.car_data.position[1]) >= self.min_car_dist_from_backboard):
                self.second_air_touch=True

        if self.off_backboard and self.second_air_touch and self.num_steps < 5:
            if player.team_num == BLUE_TEAM:
                objective = np.array(ORANGE_GOAL_BACK)
            else:
                objective = np.array(BLUE_GOAL_BACK)
            vel = state.ball.linear_velocity
            pos_diff = objective - state.ball.position
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            if np.linalg.norm(vel) > 0:
                norm_vel = vel / np.linalg.norm(vel)
            else:
                norm_vel = vel
            # only care about going towards back of net in x, y plane. z coordinate can vary.
            dot = float(norm_pos_diff[0]*norm_vel[0] + norm_pos_diff[1]*norm_vel[1])
            # check to see if velocity of ball towards goal is positive after final touch, only give reward if true
            if dot > .5:
                reward=1
                # 5x reward if get full double tap instead of just backboard read
                if self.first_air_touch:
                    reward = 5
            # increment steps to make sure only rewards for initial hit + follow up
            self.num_steps += 1

        # if ball hits ground, reset conditions and no reward
        # reset in 5 steps? Avoids cheat code of hitting off backboard then dribbling ball around on top of car to get max continual reward
        if ball_position[2] < self.min_height or self.num_steps > 5:
            self.first_air_touch=False
            self.second_air_touch=False
            self.off_backboard=False
 
        self.prev_ball_vel = state.ball.linear_velocity

        return reward



class AirDribbleReward(RewardFunction):
    """Class to reward agent for air dribbles. Gets progressively rewarded for behavior closer and closer to ideal air dribble behavior."""
    def __init__(self):
        # off_backboard and air_touch vars initialized to False. Set to true when each event occurrs, set back to false when ball touches ground
        self.wall_touch = False
        self.off_sidewall = False
        self.towards_ball = False
        self.first_air_touch = False
        self.second_air_touch = False
        self.min_wall_touch_height = 100
        self.min_air_dribble_height = 500 # top of goal is 642.775
        self.min_pos_diff = 350
        self.min_vel_diff = 400
        self.num_steps = 0
        self.max_height = 2044-92.75
        self.num_steps = 0


    def reset(self, initial_state: GameState):
        self.wall_touch = False
        self.off_sidewall = False
        self.towards_ball = False
        self.first_air_touch = False
        self.second_air_touch = False
        self.num_steps = 0
        self.num_steps = 0

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        # reward starts at 0
        reward = 0

        # need to account for attacking goal based on team
        if player.team_num == BLUE_TEAM:
            ball_position = state.ball.position
            ball_vel = state.ball.linear_velocity
        else:
            ball_position = state.inverted_ball.position
            ball_vel = state.inverted_ball.linear_velocity


        # set wall_touch to true if touching ball while car is on wall
        if player.ball_touched and player.on_ground and ball_position[2] >= self.min_wall_touch_height and abs(ball_position[1]) <= 3500 and self.num_steps < 15: # backboard is 5120
            self.wall_touch = True
            self.num_steps = self.num_steps + 1
            reward = .1*((ball_position[2] - self.min_wall_touch_height)/self.max_height)**.5 # diminshing returns, increasing rewards for higher touches on wall

        # add reward if hits ball off sidewall up into air towards goal. only true if already wall touch
        if not self.off_sidewall and self.wall_touch and ball_vel[2] > 0 and ball_vel[1] > 0 and abs(ball_position[0]) < 4096 - BALL_RADIUS*2:
            self.num_steps = 0
            self.off_sidewall = True
            reward = .25

        # add reward if car off wall towards ball
        if self.wall_touch and self.off_sidewall and not player.on_ground:
            pos_diff = state.ball.position - player.car_data.position
            vel_diff = state.ball.linear_velocity - player.car_data.linear_velocity
            norm_pos_diff = np.linalg.norm(pos_diff)
            norm_vel_diff = np.linalg.norm(vel_diff)

            # reward as long as stays close to ball and with a similar velocity as ball
            if norm_pos_diff < self.min_pos_diff and norm_vel_diff < self.min_vel_diff and self.num_steps < 30:
                self.num_steps = self.num_steps + 1
                self.towards_ball = True
                reward = .1

        # add reward if hits ball in air after pop
        if not self.first_air_touch and self.wall_touch and self.off_sidewall and self.towards_ball and player.ball_touched and not player.on_ground and ball_position[2] >=  self.min_air_dribble_height:
            self.first_air_touch = True
            reward = reward + 1.5

        # add reward for each following air touch
        if self.wall_touch and self.off_sidewall and self.towards_ball and self.first_air_touch and  player.ball_touched and not player.on_ground and ball_position[2] >=  self.min_air_dribble_height:
            self.second_air_touch = True
            reward = max(2, reward + 1.5)


        # add extra reward if ball going towards goal
        if self.wall_touch and self.off_sidewall and self.towards_ball and self.second_air_touch:
            if player.team_num == BLUE_TEAM:
                objective = np.array(ORANGE_GOAL_BACK)
            else:
                objective = np.array(BLUE_GOAL_BACK)
            vel = state.ball.linear_velocity
            pos_diff = objective - state.ball.position
            # Regular component velocity
            norm_pos_diff = pos_diff / np.linalg.norm(pos_diff)
            if np.linalg.norm(vel) > 0:
                norm_vel = vel / np.linalg.norm(vel)
            else:
                norm_vel = vel
            # only care about going towards back of net in x, y plane. z coordinate can vary.
            dot = float(norm_pos_diff[0]*norm_vel[0] + norm_pos_diff[1]*norm_vel[1])
            if dot > .75:
                reward = reward + 1


        # if ball hits ground, reset conditions and no reward
        if ball_position[2] < self.min_wall_touch_height or self.num_steps > 5:
            self.wall_touch = False
            self.off_sidewall = False
            self.towards_ball = False
            self.first_air_touch = False
            self.num_steps = 0
            reward = 0
 

        return reward


class CronusRewards(RewardFunction):
    def __init__(self):
        super().__init__()
        # defining initial event reward weights, will update over time for curriculum learning and comment iterations
        # v0.1
        self.goal_weight = 10
        self.demo_weight = 4
        self.boost_weight = .025 # from 1 from 2
        self.shot_weight=1
        self.touch_weight = 4 # from 2 from 0

        # defining initial custom reward weights, will update over time for curriculum learning and comment iterations
        # v0.1
        self.event_weight = 1
        self.touch_vel_weight = .75 # from .5
        self.align_weight = 0 # from.25 around 60 mil steps 
        self.vel_ball_weight = .05
        self.vel_weight = .0025 # from.02 # from .01 # from .025
        self.maintain_vel_weight = 0 # from .015 # from .1 from .025
        self.bad_turtle_weight = .25 # from .5
        self.jump_touch_weight = .5 # from 0
        self.double_tap_weight = 1 # from .5

        self.reward = SB3CombinedLogReward(
            (
             EventReward(goal=self.goal_weight, concede=-self.goal_weight, demo=self.demo_weight, boost_pickup=self.boost_weight, shot=self.shot_weight, touch=self.touch_weight),  
             TouchVelChange(),
             AlignBallGoal(),
             VelocityBallToGoalReward(),
             VelocityReward(),
             BadTurtle(),
             JumpTouchReward(),
             DoubleTapReward()

             ),
            (self.event_weight, self.touch_vel_weight, self.align_weight, self.vel_ball_weight, self.vel_weight, self.bad_turtle_weight, self.jump_touch_weight, self.double_tap_weight),
            "logs"
        )

    def reset(self, initial_state: GameState) -> None:
        self.reward.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.reward.get_reward(player, state, previous_action)
