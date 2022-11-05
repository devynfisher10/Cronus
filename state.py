
import random

import numpy as np

from rlgym.utils.common_values import CAR_MAX_SPEED, SIDE_WALL_X, BACK_WALL_Y, CEILING_Z, BALL_RADIUS, CAR_MAX_ANG_VEL, \
    BALL_MAX_SPEED


from numpy import random as rand

from rlgym.utils import StateSetter
from rlgym.utils.state_setters import DefaultState, StateWrapper, RandomState
from rlgym_tools.extra_state_setters.symmetric_setter import KickoffLikeSetter
from rlgym_tools.extra_state_setters.goalie_state import GoaliePracticeState
from rlgym_tools.extra_state_setters.wall_state import WallPracticeState

LIM_X = SIDE_WALL_X - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Y = BACK_WALL_Y - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Z = CEILING_Z - BALL_RADIUS

PITCH_LIM = np.pi / 2
YAW_LIM = np.pi
ROLL_LIM = np.pi

GOAL_X_MAX = 800.0
GOAL_X_MIN = -800.0

PLACEMENT_BOX_X = 5000
PLACEMENT_BOX_Y = 2000
PLACEMENT_BOX_Y_OFFSET = 3000

GOAL_LINE = 5100

YAW_MAX = np.pi




# goal is to use all the extra tools state setters and a random one + maybe write one of my own (aerial shot state?) (air dribble / reset state?) 
# then give each a certain prob


class AerialPracticeState(StateSetter):

    def __init__(self, reset_to_max_boost=True):
        """
        AerialPracticeState constructor.
        :param reset_to_max_boost: Boolean indicating whether the cars will start each episode with 100 boost or keep from last episode
        """
        super().__init__()
        self.team_turn = 0  # swap every reset which car is making the aerial play
        self.reset_to_max_boost = reset_to_max_boost

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies the StateWrapper to set a new aerial play
        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        self._reset_ball_and_cars(state_wrapper, self.team_turn, self.reset_to_max_boost)

        # which team will make the next aerial play
        self.team_turn = (self.team_turn + 1) % 2

    def _place_car_in_box_area(self, car, team_delin):
        """
        Function to place a car in an allowed area
        :param car: car to be modified
        :param team_delin: team number delinator to look at when deciding where to place the car
        """

        y_pos = (PLACEMENT_BOX_Y - (rand.random() * PLACEMENT_BOX_Y))

        if team_delin == 0:
            y_pos -= PLACEMENT_BOX_Y_OFFSET
        else:
            y_pos += PLACEMENT_BOX_Y_OFFSET

        car.set_pos(rand.random() * PLACEMENT_BOX_X - PLACEMENT_BOX_X / 2, y_pos, z=17)

    def _reset_ball_and_cars(self, state_wrapper: StateWrapper, team_turn, reset_to_max_boost):
        """
        Function to set a new ball in the air towards a goal
        :param state_wrapper: StateWrapper object to be modified.
        :param team_turn: team who's making the aerial play
        :param reset_to_max_boost: Boolean indicating whether the cars will start each episode with 100 boost or keep from last episode
        """

        # reset ball
        pos, lin_vel, ang_vel = self._get_ball_parameters(team_turn)
        state_wrapper.ball.set_pos(pos[0], pos[1], pos[2])
        state_wrapper.ball.set_lin_vel(lin_vel[0], lin_vel[1], lin_vel[2])
        state_wrapper.ball.set_ang_vel(ang_vel[0], ang_vel[1], ang_vel[2])

        # reset cars relative to the ball
        first_set = False
        for car in state_wrapper.cars:
            # set random position and rotation for all cars based on pre-determined ranges

            if car.team_num == team_turn and not first_set:
                car.set_pos(x=pos[0] * random.uniform(.9, 1.1), y=pos[1] * random.uniform(.9, 1.1),  z=pos[2] * random.uniform(.9, 1.1))
                car.set_lin_vel(lin_vel[0] * random.uniform(.9, 1.1), lin_vel[1] * random.uniform(.9, 1.1), lin_vel[2] * random.uniform(.9, 1.1))
                first_set = True
            else:
                self._place_car_in_box_area(car, car.team_num)

            if reset_to_max_boost:
                car.boost = 100

            car.set_rot(0, rand.random() * YAW_MAX - YAW_MAX / 2, 0)


    def _get_ball_parameters(self, team_turn):
        """
        Function to set a new ball up for an aerial play
        
        :param team_turn: team who's making the aerial play
        """

        INVERT_IF_BLUE = (-1 if team_turn == 0 else 1)  # invert shot for blue

        # set positional values
        x_pos = random.uniform(GOAL_X_MIN, GOAL_X_MAX)
        y_pos = 1000 * random.uniform(-.1, 1) * INVERT_IF_BLUE
        z_pos = CEILING_Z * random.uniform(.5, 1)
        pos = np.array([x_pos, y_pos, z_pos])

        # set lin velocity values
        x_vel_randomizer = (random.uniform(-.5, .5))
        y_vel_randomizer = (random.uniform(-.1, .3))
        z_vel_randomizer = (random.uniform(.1, 1))

        x_vel = (200 * x_vel_randomizer)
        y_vel = (200 * y_vel_randomizer * INVERT_IF_BLUE)
        z_vel = (600 * z_vel_randomizer)
        lin_vel = np.array([x_vel, y_vel, z_vel])


        ang_vel = np.array([0, 0, 0])

        return pos, lin_vel, ang_vel




class TurtleState(StateSetter):

    def __init__(self, reset_to_max_boost=True):
        """
        AerialPracticeState constructor.
        :param reset_to_max_boost: Boolean indicating whether the cars will start each episode with 100 boost or keep from last episode
        """
        super().__init__()
        self.team_turn = 0  # swap every reset which car is making the aerial play
        self.reset_to_max_boost = reset_to_max_boost

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies the StateWrapper to set a new aerial play
        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        self._reset_ball_and_cars(state_wrapper, self.team_turn, self.reset_to_max_boost)

        # which team will make the next aerial play
        self.team_turn = (self.team_turn + 1) % 2

    def _reset_ball_and_cars(self, state_wrapper: StateWrapper, team_turn, reset_to_max_boost):
        """
        Function to set a new ball in the air towards a goal
        :param state_wrapper: StateWrapper object to be modified.
        :param team_turn: team who's making the aerial play
        :param reset_to_max_boost: Boolean indicating whether the cars will start each episode with 100 boost or keep from last episode
        """

        pos, lin_vel, ang_vel = self._get_ball_parameters(team_turn)
        state_wrapper.ball.set_pos(pos[0], pos[1], pos[2])
        state_wrapper.ball.set_lin_vel(lin_vel[0], lin_vel[1], lin_vel[2])
        state_wrapper.ball.set_ang_vel(ang_vel[0], ang_vel[1], ang_vel[2])

        # reset cars relative to the ball
        first_set = False
        for car in state_wrapper.cars:
            # set random position and rotation for all cars based on pre-determined ranges

            if car.team_num == team_turn and not first_set:
                car.set_pos(x=1000, y=4000,  z=40.24)
                car.set_lin_vel(0,0,0)
                first_set = True
            else:
                car.set_pos(x=-1000, y=-4000,  z=40.24)
                car.set_lin_vel(0,0,0)

            car.boost = 100

            car.quaternion = (1, 0, 0, -1)

            car.set_rot(0, 0, 0)


    def _get_ball_parameters(self, team_turn):
        """
        Function to set a new ball up for an aerial play
        
        :param team_turn: team who's making the aerial play
        """

        INVERT_IF_BLUE = (-1 if team_turn == 0 else 1)  # invert shot for blue

        # set positional values
        x_pos = random.uniform(GOAL_X_MIN, GOAL_X_MAX)
        y_pos = 1000 * random.uniform(-.1, 1) * INVERT_IF_BLUE
        z_pos = CEILING_Z * random.uniform(.5, 1)
        pos = np.array([x_pos, y_pos, z_pos])

        # set lin velocity values
        x_vel_randomizer = (random.uniform(-.5, .5))
        y_vel_randomizer = (random.uniform(-.1, .3))
        z_vel_randomizer = (random.uniform(.1, 1))

        x_vel = (200 * x_vel_randomizer)
        y_vel = (200 * y_vel_randomizer * INVERT_IF_BLUE)
        z_vel = (600 * z_vel_randomizer)
        lin_vel = np.array([x_vel, y_vel, z_vel])


        ang_vel = np.array([0, 0, 0])

        return pos, lin_vel, ang_vel


# setting initial probabilities for each state here. Will update over time as increase prob of complex states for curriculum learning.
class CronusStateSetter(StateSetter):
    def __init__(
            self, *,
            goalie_prob=0.0, 
            wall_prob=0.25,
            default_prob=.2,
            kickofflike_prob=0.1,
            random_prob=.35,
            aerial_prob=.1
    ):


        super().__init__()


        self.setters = [
            GoaliePracticeState(),
            WallPracticeState(air_dribble_odds=7/10, backboard_roll_odds=2/10, side_high_odds=1/10),
            DefaultState(),
            KickoffLikeSetter(),
            RandomState(),
            AerialPracticeState(),
        ]
        self.probs = np.array([goalie_prob, wall_prob, default_prob, kickofflike_prob, random_prob, aerial_prob])

        """
        self.setters = [
            TurtleState(),
        ]
        self.probs = np.array([1])
        """

        assert self.probs.sum() == 1, "Probabilities must sum to 1"

    def reset(self, state_wrapper: StateWrapper):
        i = np.random.choice(len(self.setters), p=self.probs)
        self.setters[i].reset(state_wrapper)