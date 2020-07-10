import gym
from gym import spaces
import numpy as np


class ImagePathEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ImagePathEnv, self).__init__()
        self.reward_range = (0, 1)
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([3]), dtype=np.float16)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10, 10), dtype=np.float16)
        self.current_step = 0
        self.viewer = None
        self.state = None
        self.agent_position = None
        self.goal_position = None

    def _next_observation(self):
        obs = np.zeros(self.observation_space.shape)
        obs[self.agent_position[0], self.agent_position[1]] = 1
        obs[self.goal_position[0], self.goal_position[1]] = 1
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
        self._take_action(action)
        self.current_step += 1
        reward = 0
        if self.goal_position == self.agent_position:
            reward = 1
        else:
            reward = -1

        done = bool(self.goal_position == self.agent_position
                    or self.agent_position[0] < 0
                    or self.agent_position[0] > self.observation_space.shape[0]
                    or self.agent_position[1] < 0
                    or self.agent_position[1] > self.observation_space.shape[1]
                    or self.current_step > 20
                    )

        obs = None
        if not done:
            obs = self._next_observation()
        self.state = obs
        return obs, reward, done

    def reset(self):
        self.agent_position = [0, 0]
        self.goal_position = [4, 4]
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
            self.cells = [[None for i in range(self.observation_space.shape[0])] for j in range(self.observation_space.shape[1])]
            self.viewer = rendering.Viewer(screen_width, screen_height)
            for i in range(self.observation_space.shape[0]):
                for j in range(self.observation_space.shape[1]):
                    l, r, b, t = i * cell_width, (i + 1) * cell_width - 1, j * cell_height, (j + 1) * cell_height - 1
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
                else:
                    cell.set_color(0.5, 0.5, 0.5)
                self.viewer.add_geom(cell)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None