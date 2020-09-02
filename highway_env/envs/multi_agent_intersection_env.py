from typing import Dict, Tuple

from gym.envs.registration import register
import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, CircularLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.controller import MDPVehicle
from highway_env.envs.common.observation import observation_factory, ObservationType
from highway_env.envs.common.action import action_factory, Action, DiscreteMetaAction, ActionType

from typing import List, Tuple, Optional, Callable


class MAIntersectionEnv(AbstractEnv):
    COLLISION_REWARD: float = -5
    HIGH_SPEED_REWARD: float = 1
    ARRIVED_REWARD: float = 1

    ACTIONS: Dict[int, str] = {
        0: 'SLOWER',
        1: 'IDLE',
        2: 'FASTER'
    }
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        # for list of features -> it sorts vehicles in the observation field and return a matrix. If the number of agents in the obs field is less than vehicle_counts, it does zero padding 
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 15,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "absolute": True,
                "flatten": False,
                "observe_intentions": False,
            },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": False
            },
            "duration": 13,  # [s]
            "initial_vehicle_count": 10,
            "spawn_probability": 0.6,
            "screen_width": 1000,
            "screen_height": 1000,
            "centering_position": [0.5, 0.5],
            "scaling": 5.5 * 1.3,
            "collision_reward": MAIntersectionEnv.COLLISION_REWARD,
            "normalize_reward": False,
            "controlled_vehicles_count": 2,
            "simulation_frequency": 15,  # [Hz]
            "auto_select_starts_ends": True, #if false, set start and end indexes manually
            "start_positions": [0, 2],
            "end_positions": [1, 0]
        })

        ## for occupancy grid obs
        # config.update({
        #     "observation": {
        #         "type": "OccupancyGrid",
        #         "grid_step": [1, 1],
        #         "vehicles_count": 15,
        #         "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        #         "features_range": {
        #             "x": [-100, 100],
        #             "y": [-100, 100],
        #             "vx": [-20, 20],
        #             "vy": [-20, 20],
        #         },
        #         "absolute": False,
        #         "flatten": False,
        #         "observe_intentions": False,
        #     },
        #     "action": {
        #         "type": "DiscreteMetaAction",
        #         "longitudinal": True,
        #         "lateral": False
        #     },
        #     "duration": 300,  # [s]
        #     "destination": "o1",
        #     "initial_vehicle_count": 10,
        #     "spawn_probability": 0.6,
        #     "screen_width": 1000,
        #     "screen_height": 1000,
        #     "centering_position": [0.5, 0.5],
        #     "scaling": 5.5 * 1.3,
        #     "collision_reward": MAIntersectionEnv.COLLISION_REWARD,
        #     "normalize_reward": False,
        #     "controlled_vehicles_count": 2,
        #     "simulation_frequency": 15,  # [Hz]
            # "auto_select_starts_ends": True, #if false, set start and end indexes manually
            # "start_positions": [0, 2],
            # "end_positions": [1, 0],
            # "spawn_random_vehicles": True
        # })
        
        return config

    def define_spaces(self) -> None:
        self.observation_type = observation_factory(self, self.config["observation"])
        self.observation_space = [self.observation_type.space() for _ in range(self.config['controlled_vehicles_count'])]
        self.action_type = action_factory(self, self.config["action"])
        self.action_space = [self.action_type.space() for _ in range(self.config['controlled_vehicles_count'])]

    def _reward(self, action: List[int]) -> List[float]:
        reward_list = []
        for v, a in zip(self.controlled_vehicles, action):
            reward = self.config["collision_reward"] * v.crashed \
                 + self.HIGH_SPEED_REWARD * (v.speed_index == v.SPEED_COUNT - 1)
            reward = self.ARRIVED_REWARD if self.has_arrived(v) else reward
            if self.config["normalize_reward"]:
                reward = utils.lmap(reward, [self.config["collision_reward"], self.ARRIVED_REWARD], [0, 1])
            reward_list.append(reward)
        return reward_list

    def _get_obs(self) -> List[np.ndarray]:
        """
        Calculates observation for all controlled vehicles.

        :return: a dict
        """
        obs = []
        for v in self.controlled_vehicles:
            obs.append(self.observation_type.observe(v))
        return obs

    def reset(self) -> List[np.ndarray]:
        """
        Reset the environment to it's initial configuration

        :return: the observation of the reset state
        """
        self.time = 0
        self.controlled_vehicles = []
        self.done = [False for _ in range(len(self.controlled_vehicles))]
        self.define_spaces()
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])
        self.steps = 0
        return self._get_obs()

    def _simulate(self, action: Optional[Action] = None) -> None:
        """Perform several steps of simulation with constant action."""
        for _ in range(int(self.config["simulation_frequency"] // self.config["policy_frequency"])):
            # Forward action to the vehicle
            if action is not None \
                    and not self.config["manual_control"] \
                    and self.time % int(self.config["simulation_frequency"] // self.config["policy_frequency"]) == 0:
                self.action_type.act(action)

            self.road.act()
            self.road.step(1 / self.config["simulation_frequency"])
            self.time += 1

            # Automatically render intermediate simulation steps if a viewer has been launched
            # Ignored if the rendering is done offscreen
            self._automatic_rendering()

            # Stop at terminal states
            if self.done or self._is_terminal():
                break
        self.enable_auto_render = False

    def step(self, action: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], dict]:
        """
        Perform an action and step the environment dynamics.

        The action is executed by the ego-vehicle, and all other vehicles on the road performs their default behaviour
        for several simulation timesteps until the next decision making step.

        :param action: the action performed by the ego-vehicle
        :return: a tuple (observation, reward, terminal, info)
        """
        if self.road is None or len(self.controlled_vehicles) == 0:
            raise NotImplementedError("The road and vehicle must be initialized in the environment implementation")

        self._simulate(action)

        obs = self._get_obs()
        reward = self._reward(action)
        terminal = self._is_terminal()

        info = {
            "speed": [v.speed for v in self.controlled_vehicles],
            "crashed": [v.crashed for v in self.controlled_vehicles],
            "action": action,
        }
        try:
            info["cost"] = self._cost(action)
        except NotImplementedError:
            pass

        # return obs, reward, terminal, info

        self.steps += 1
        self._clear_vehicles()
        self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])
        return (obs, reward, terminal, info)

    def _is_terminal(self) -> List[bool]:
        """The episode is over when episode duration is finished"""

        # dones = []
        # for i, v in enumerate(self.controlled_vehicles):
        #     done = v.crashed \
        #         or self.steps >= self.config["duration"] * self.config["policy_frequency"] \
        #         or self.has_arrived(v)
        #     if done:
        #         print(f"agent={i}, crashed={v.crashed}, arrived={self.has_arrived(v)}" + ', step=', self.steps >= self.config["duration"] * self.config["policy_frequency"])
        #     dones.append(done)

        dones = [self.steps >= self.config["duration"] * self.config["policy_frequency"]] * self.config["controlled_vehicles_count"]
        return dones

    def _make_road(self) -> None:
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Incoming
            start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane("o" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=[s, c], priority=priority, speed_limit=10))
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane("ir" + str(corner), "il" + str((corner - 1) % 4),
                         CircularLane(r_center, right_turn_radius, angle + np.radians(180), angle + np.radians(270),
                                      line_types=[n, c], priority=priority, speed_limit=10))
            # Left turn
            l_center = rotation @ (np.array([-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2]))
            net.add_lane("ir" + str(corner), "il" + str((corner + 1) % 4),
                         CircularLane(l_center, left_turn_radius, angle + np.radians(0), angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane("ir" + str(corner), "il" + str((corner + 2) % 4),
                         StraightLane(start, end, line_types=[s, n], priority=priority, speed_limit=10))
            # Exit
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane("il" + str((corner - 1) % 4), "o" + str((corner - 1) % 4),
                         StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10))

        road = RegulatedRoad(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        self.road = road

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        if self.config["spawn_probability"] > 0:
            simulation_steps = 3
            for t in range(n_vehicles - 1):
                self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
            for _ in range(simulation_steps):
                [(self.road.act(), self.road.step(1 / self.config["simulation_frequency"])) for _ in range(self.config["simulation_frequency"])]

            # Challenger vehicle
            self._spawn_vehicle(60, spawn_probability=1, go_straight=True, position_deviation=0.1, speed_deviation=0)

        # Ego-vehicle
        starts = []
        dests = []
        irs = []
        
        for n in range(self.config['controlled_vehicles_count']):
            if self.config["auto_select_starts_ends"]:
                while True:
                    s = np.random.randint(0, 4)
                    if "o"+str(s) not in starts:
                        break
                while True: #select an end which is not equal to start
                    e = np.random.randint(0, 4)
                    if e != s and "o"+str(e) not in dests:
                        break
            else:
                s = self.config["start_positions"][n]
                e = self.config["end_positions"][n]
            starts.append("o"+str(s))
            dests.append("o"+str(e))
            irs.append("ir"+str(s))

        # lerning_vehicles = []
        for ego_i in range(self.config['controlled_vehicles_count']):
            # ego_lane = self.road.network.get_lane(("o0", "ir0", 0))
            # destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
            ego_lane = self.road.network.get_lane((starts[ego_i], irs[ego_i], 0))
            destination = dests[ego_i]
            ego_vehicle = self.action_type.vehicle_class(
                         self.road,
                         ego_lane.position(60, 0),
                         speed=ego_lane.speed_limit,
                         heading=ego_lane.heading_at(50)) \
                .plan_route_to(destination)
            ego_vehicle.SPEED_MIN = 0
            ego_vehicle.SPEED_MAX = 9
            ego_vehicle.SPEED_COUNT = 3
            ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
            ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)

            # ego_vehicle.TAU_A = 1.0
            self.road.vehicles.append(ego_vehicle)
            # learning_vehicles.append(ego_vehicle)
            # self.vehicle = ego_vehicle
            self.controlled_vehicles.append(ego_vehicle)

        for v in self.road.vehicles:  # Prevent early collisions
            if v not in self.controlled_vehicles:
                cond = [np.linalg.norm(v.position - controlled_vehicle.position) < 20 for controlled_vehicle in self.controlled_vehicles]
                if sum(cond):
                    self.road.vehicles.remove(v)

    def _spawn_vehicle(self,
                       longitudinal: float = 0,
                       position_deviation: float = 1.,
                       speed_deviation: float = 1.,
                       spawn_probability: float = 0.6,
                       go_straight: bool = False) -> None:
        if self.np_random.rand() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(self.road, ("o" + str(route[0]), "ir" + str(route[0]), 0),
                                            longitudinal=longitudinal + 5 + self.np_random.randn() * position_deviation,
                                            speed=8 + self.np_random.randn() * speed_deviation)
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        is_leaving = lambda vehicle: "il" in vehicle.lane_index[0] and "o" in vehicle.lane_index[1] \
                                     and vehicle.lane.local_coordinates(vehicle.position)[0] \
                                     >= vehicle.lane.length - 4 * vehicle.LENGTH
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle is self.vehicle or not (is_leaving(vehicle) or vehicle.route is None)]

    def has_arrived(self, vehicle, exit_distance=25) -> bool:
        return "il" in vehicle.lane_index[0] \
               and "o" in vehicle.lane_index[1] \
               and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance

    def _cost(self, action: int) -> List[float]:
        """The constraint signal is the occurrence of collisions."""
        return [float(v.crashed) for v in self.controlled_vehicles]
    

register(
    id='intersection-v1',
    entry_point='highway_env.envs:MAIntersectionEnv',
)
