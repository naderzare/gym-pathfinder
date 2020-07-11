import gym
from gym import spaces
import numpy as np
import random


class PathFinderEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PathFinderEnv, self).__init__()
        self.sparse_reward = False
        self.count_i = 10
        self.count_j = 10
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([3]), dtype=np.float16)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.count_i, self.count_j), dtype=np.float16)
        self.current_step = 0
        self.viewer = None
        self.state = None
        self.agent_position = None
        self.goal_position = None
        self.walls = None

    def _next_observation(self):
        obs = np.zeros(self.observation_space.shape)
        obs[self.agent_position[0], self.agent_position[1]] = 1
        obs[self.goal_position[0], self.goal_position[1]] = 0.5
        for w in self.walls:
            obs[w[0], w[1]] = 0.25
        return obs

    def _take_action(self, action):
        if action == 0:
            self.agent_position[0] += 1
        if action == 2:
            self.agent_position[0] -= 1
        if action == 1:
            self.agent_position[1] -= 1
        if action == 3:
            self.agent_position[1] += 1

    def step(self, action):
        previous_position = [self.agent_position[0], self.agent_position[1]]
        self._take_action(action)
        self.current_step += 1
        reward = 0
        if self.sparse_reward:
            if self.goal_position == self.agent_position:
                reward = 1
            else:
                reward = -1
        else:
            if self.goal_position == self.agent_position:
                reward = 1
            elif self.agent_position[0] < 0 \
                    or self.agent_position[0] > self.observation_space.shape[0] \
                    or self.agent_position[1] < 0 or self.agent_position[1] > self.observation_space.shape[1] \
                    or self.agent_position in self.walls:
                reward = -1
            else:
                c_diff = abs(self.agent_position[0] - self.goal_position[0]) + abs(
                    self.agent_position[1] - self.goal_position[1])
                p_diff = abs(previous_position[0] - self.goal_position[0]) + abs(
                    previous_position[1] - self.goal_position[1])
                reward = (p_diff - c_diff) / 20

        done = bool(self.goal_position == self.agent_position
                    or self.agent_position[0] < 0
                    or self.agent_position[0] > self.observation_space.shape[0]
                    or self.agent_position[1] < 0
                    or self.agent_position[1] > self.observation_space.shape[1]
                    or self.current_step > 20
                    or self.agent_position in self.walls
                    )

        obs = None
        if not done:
            obs = self._next_observation()
        self.state = obs
        return obs, reward, done

    def reset(self):
        wall_inserted = 0
        self.walls = []
        while wall_inserted < 5:
            wall = [random.randint(0, self.count_i - 1), random.randint(0, self.count_j - 1)]
            if wall not in self.walls:
                self.walls.append(wall)
                wall_inserted += 1
        while True:
            self.agent_position = [random.randint(0, self.count_i - 1), random.randint(0, self.count_j - 1)]
            if self.agent_position not in self.walls:
                break
        while True:
            self.goal_position = [random.randint(0, self.count_i - 1), random.randint(0, self.count_j - 1)]
            if self.goal_position != self.agent_position and self.goal_position not in self.walls:
                break
        self.current_step = 0
        self.state = np.zeros(self.observation_space.shape)
        obs = self._next_observation()
        self.state = obs
        return obs

    def render(self, mode='human', close=False):
        screen_width = 400
        screen_height = 400
        cell_width = screen_width / self.observation_space.shape[0]
        cell_height = screen_height / self.observation_space.shape[1]
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.cells = [[None for i in range(self.observation_space.shape[0])] for j in
                          range(self.observation_space.shape[1])]
            self.viewer = rendering.Viewer(screen_width, screen_height)
            for i in range(self.observation_space.shape[1]):
                for j in range(self.observation_space.shape[0]):
                    l, r, b, t = j * cell_width, \
                                 (j + 1) * cell_width - 1, \
                                 (self.observation_space.shape[0] - i - 1) * cell_height, \
                                 (self.observation_space.shape[0] - i) * cell_height - 1
                    cell = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
                    cell.set_color(0.5, 0.5, 0.5)
                    self.cells[i][j] = cell
                    self.viewer.add_geom(cell)

        for i in range(self.observation_space.shape[0]):
            for j in range(self.observation_space.shape[1]):
                cell = self.cells[i][j]
                if self.agent_position == [i, j]:
                    cell.set_color(0, 0, 1)
                elif self.goal_position == [i, j]:
                    cell.set_color(0, 1, 0)
                elif [i, j] in self.walls:
                    cell.set_color(0., 0., 0.)
                else:
                    cell.set_color(0.5, 0.5, 0.5)
                self.viewer.add_geom(cell)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
