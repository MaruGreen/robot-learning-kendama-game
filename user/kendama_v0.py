'''

Author: Li Shidi
Date: Jun, 2019
Mail address: E0081728@u.nus.edu
National University of Singapore

This is a simulator for robot playing Kendama manipulation. For more information, please refer to the paper:
Robot Trajectory Learning Framework for Precise Tasks with Discontinuous Dynamics and High Speed.

'''

import gym
import numpy as np

class KendamaEnv0(gym.Env):

    def __init__(self):
        '''
        The 13 observations are (ken_y, ken_z, ken_vy, ken_vz, theta, omega,
                                dama_x, dama_z, dama_vy, dama_vz, dama_theta, dama_omega, T)
        While T > 0, 13 observations are dependent to the first 6 states.
        When T = 0,
        '''
        self.observation = np.zeros(13)
        self.observation_space = self.observation.shape
        self.action_space = np.zeros(2).shape

        self.mass = 1.0          # unit mass
        self.length = 400.0      # millimeter
        self.kenlength = 30.0    # millimeter
        self.kencorner = (25.0, 15.0)
        self.damaradius = 35.0   # millimeter
        self.gravity = 9780.0    # mm / s2
        self.friction = 0.825
        self.dt = 0.01           # second
        self.idealend = 67       # degree
        self.gamma = [1, 1, 1]
        self.alpha = [5e-3, 1e-4, 5e-3]
        self.tighten = True
        self.finalized = False
        self.collision = False
        self.success = False
        self.reference = []
        self.beta = np.pi * 5 / 6

        self.viewer = None

    def _immediate_reward(self, obs, act):
        '''
        L(st, at) = 0.5 * (st - st^ref).T * Q * (st - st^ref) + 0.5 * (at - at^ref).T * R * (at - at^ref)
        For st, consider only the first 4 elements
        Immediate rewards should be further normalized due to different step length
        '''
        reward = 0
        num_traj_count = 0
        for traj in self.reference:
            if len(traj['actions']) <= self.time_step:
                continue
            num_traj_count += 1
            ref_obs = traj['observations'][self.time_step][0:4].reshape([1, 4])
            ref_act = traj['actions'][self.time_step].reshape([1, 2])
            cur_obs = obs[0:4].copy().reshape([1, 4])
            cur_act = act.copy().reshape([1, 2])
            reward += 0.5 * np.matrix(ref_obs - cur_obs) * traj['Q_matrix'] * np.matrix(ref_obs - cur_obs).T
            reward += 0.5 * np.matrix(ref_act - cur_act) * traj['R_matrix'] * np.matrix(ref_act - cur_act).T
        if num_traj_count == 0:
            return 0
        return float(reward) / num_traj_count

    def _final_reward(self, obs):
        # part 1: check if the Dama hole is facing the movement forward direction
        theta = obs[10] - np.pi / 2
        phi = np.pi + np.arctan2(obs[9], obs[8])
        angle_margin = (theta - phi) * 180 / np.pi

        # part 2: check the distance error
        dis = (obs[6] - obs[0]) - (obs[7] - obs[1]) / np.tan(phi)

        # part 3: the ideal situation is theta = phi = 67 degree
        # phi will not change much, usually within 50 to 90 degree
        # but theta can be from 0 to 180
        # how about letting part 3 focus on theta, and letting part 1 affect phi
        theta_margin = theta * 180 / np.pi - self.idealend

        print('The distance error is', dis, 'mm')
        print('The Dama hole direciton is', theta * 180 / np.pi, 'degree')
        print('The Dama movement direction is', phi * 180 / np.pi, 'degree')
        # the success conditions
        if np.abs(angle_margin) < 10 and np.power(dis, 2) < 30 and np.abs(theta_margin) < 15:
        #if np.abs(angle_margin) < 30 and np.power(dis, 2) < 30 and np.abs(theta_margin) < 30:
            print('Ken catches Dama successfully!')
            self.success = True

        # combine 3 parts for the final reward
        r1 = np.exp(-self.alpha[0] * np.power(angle_margin, 2))
        r2 = np.exp(-self.alpha[1] * np.power(dis, 2))
        r3 = np.exp(-self.alpha[2] * np.power(theta_margin, 2))
        reward = (self.gamma[0] * r1 + self.gamma[1] * r2 + self.gamma[2] * r3) / sum(self.gamma)
        print('The final reward is', reward)

        self.finalized = True
        return reward

    def setReference(self, traj):
        self.reference.append(traj)

    def setParameter(self, dict):
        self.mass = dict['mass']
        self.length = dict['length']
        self.kenlength = dict['kenlength']
        self.kencorner = dict['kencorner']
        self.damaradius = dict['damaradius']
        self.gravity = dict['gravity']
        self.friction = dict['friction']
        self.dt = dict['dt']
        self.idealend = dict['idealend']
        self.gamma = dict['gamma']
        self.alpha = dict['alpha']

    def _check_collision(self, obs):
        reward = 0
        # calculate the coordinates for ken pin and ken corner
        sin = np.sin(self.beta - np.pi / 2)
        cos = np.cos(self.beta - np.pi / 2)
        pin_x = obs[0] + self.kenlength * cos
        pin_y = obs[1] + self.kenlength * sin
        cor_x = obs[0] + self.kencorner[1] * cos + self.kencorner[0] * sin
        cor_y = obs[1] + self.kencorner[1] * sin - self.kencorner[0] * cos

        dis_to_pin = np.power(obs[6] - pin_x, 2) + np.power(obs[7] - pin_y, 2)
        dis_to_cor = np.power(obs[6] - cor_x, 2) + np.power(obs[7] - cor_y, 2)
        if dis_to_pin < np.power(self.damaradius, 2) and not self.finalized:
            self.collision = True
            reward += self._final_reward(obs)
            if not self.success:
                print('Collision with Ken!')
                obs[8] = -obs[8] * 0.8
                obs[9] = -obs[9] * 0.8
        elif dis_to_cor < np.power(self.damaradius, 2) and not self.finalized:
            self.collision = True
            reward += self._final_reward(obs)
            print('Collision with handle!')
            obs[8] = -obs[8] * 0.8
            obs[9] = -obs[9] * 0.8

        return reward

    def step(self, action):
        next_obs = self.observation.copy()
        reward = 0
        if self.tighten:
            # calculate the transition of the Ken
            next_obs[0] += self.observation[2] * self.dt
            next_obs[1] += self.observation[3] * self.dt
            next_obs[2] += action[0] * self.dt
            next_obs[3] += action[1] * self.dt
            next_obs[4] += self.observation[5] * self.dt
            temp = -self.friction * self.dt / self.length
            # use the old theta to update omega
            sin = np.sin(self.observation[4])
            cos = np.cos(self.observation[4])
            next_obs[5] += temp * ((self.gravity + action[1]) * sin + action[0] * cos)
            # now use the new theta to calculate the dama position
            sin = np.sin(next_obs[4])
            cos = np.cos(next_obs[4])

            # from current Ken state calculate current Dama state and Tension
            next_obs[6] = next_obs[0] + self.length * sin
            next_obs[7] = next_obs[1] - self.length * cos
            next_obs[8] = next_obs[2] + next_obs[5] * self.length * cos
            next_obs[9] = next_obs[3] + next_obs[5] * self.length * sin
            next_obs[10] = next_obs[4]
            next_obs[11] = next_obs[5]
            next_obs[12] = ((action[1] + self.gravity) * cos - action[0] * sin
                           + np.square(next_obs[5]) * self.length) * self.mass

            # use the immediate reward to constraint the new trajectory
            reward += self._immediate_reward(self.observation, action)
            # determine the tightening state
            if next_obs[12] <= 0:
                self.tighten = False
        elif self.success:
            # Ken is still under control of robot
            next_obs[0] += self.observation[2] * self.dt
            next_obs[1] += self.observation[3] * self.dt
            next_obs[2] += action[0] * self.dt
            next_obs[3] += action[1] * self.dt
            next_obs[4] = None
            next_obs[5] = None

            # from current Ken state calculate current Dama state
            next_obs[6] = next_obs[0] + self.damaradius * np.cos(self.beta - np.pi / 2)
            next_obs[7] = next_obs[1] + self.damaradius * np.sin(self.beta - np.pi / 2)
            next_obs[8] = next_obs[3]
            next_obs[9] = next_obs[4]
            next_obs[10] = self.beta
            next_obs[11] = None
            next_obs[12] = None

            reward = self._immediate_reward(self.observation, action)
        else:
            # Ken is still under control of robot
            next_obs[0] += self.observation[2] * self.dt
            next_obs[1] += self.observation[3] * self.dt
            next_obs[2] += action[0] * self.dt
            next_obs[3] += action[1] * self.dt
            next_obs[4] = None
            next_obs[5] = None

            # Dama state will be calculated by self.observation[6:12]
            next_obs[6] += self.observation[8] * self.dt
            next_obs[7] += self.observation[9] * self.dt
            next_obs[9] += -self.gravity * self.dt
            next_obs[10] += self.observation[11] * self.dt
            next_obs[12] = None

            reward = self._immediate_reward(self.observation, action)
            # make the ken rotate automatically to track the Dama hole
            if self.beta > next_obs[10]:
                self.beta -= self.observation[11] * self.dt
            elif self.beta <= next_obs[10] and next_obs[10] <= np.pi:
                self.beta = next_obs[10]
            else:
                self.beta = np.pi
            # determine if the Dama passed by Ken (for non-collision cases)
            if next_obs[1] > next_obs[7] and not self.finalized:
                reward += self._final_reward(next_obs)

        # check if collision happens or ken is going to catch dama
        if not self.collision:
            # if a collision happens, final reward will be calcualated and self.finalized will be True
            reward += self._check_collision(next_obs)

        # save the next obs to the current obs
        self.observation = next_obs

        # when crossing happens and string is about to tightening again, done
        if self.finalized and (np.square(next_obs[0] - next_obs[6]) + np.square(next_obs[1] - next_obs[7])
                           > 0.8 * np.square(self.length)):
            done = True
        else:
            done = False

        info = {}
        self.time_step += 1
        return next_obs, reward, done, info

    def reset(self, init_state=np.zeros(6)):
        self.tighten = True
        self.finalized = False
        self.collision = False
        self.success = False
        self.beta = np.pi * 5 / 6
        self.time_step = 0

        self.observation[0:13] = np.zeros(13)
        # set the initial position and velocity of Ken
        self.observation[0:6] = init_state
        sin = np.sin(self.observation[4])
        cos = np.cos(self.observation[4])

        # set the initial position and velocity of Dama
        self.observation[6] = self.observation[0] + self.length * sin
        self.observation[7] = self.observation[1] - self.length * cos
        self.observation[8] = self.observation[2] + self.observation[5] * self.length * cos
        self.observation[9] = self.observation[3] + self.observation[5] * self.length * sin
        self.observation[10] = self.observation[4]
        self.observation[11] = self.observation[5]

        # initial tension
        self.observation[12] = (self.gravity * cos + np.square(self.observation[5]) * self.length) * self.mass
        return self.observation

    def render(self, mode='human'):
        self.screen_width = 1200
        self.screen_height = 950
        self.y_left = -450
        self.y_right = 700
        self.z_bottom = -110
        world_width = self.y_right - self.y_left
        scale = self.screen_width / world_width

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

            # black background
            self.background = rendering.make_polygon([(0, 0), (0, self.screen_height), (self.screen_width, self.screen_height),
                                                      (self.screen_width, 0)], filled=True)
            self.background.set_color(0.05, 0.05, 0.05)

            # create Ken
            self.ken = rendering.make_capsule(self.kenlength * scale, self.damaradius * 0.25 * scale)
            self.kentrans = rendering.Transform()
            self.ken.add_attr(self.kentrans)
            self.ken.set_color(0, 1, 0) # red greed blue (0-black, 1-white)
            # use a polygon as the handle
            cx, cy = self.kencorner
            cx *= scale
            cy *= scale
            vec = [(0.0, 8.0 * scale), (cx, cy), (cx, -cy), (5.0 * scale, -7.5 * scale),
                   (7.0 * scale, -37.5 * scale), (9.0 * scale, -47.5 * scale), (13.0 * scale, -56 * scale),
                   (14.0 * scale, -66 * scale), (-14.0 * scale, -66 * scale), (-13.0 * scale, -56 * scale),
                   (-9.0 * scale, -47.5 * scale), (-7.0 * scale, -37.5 * scale),
                   (-5.0 * scale, -7.5 * scale), (-cx, -cy), (-cx, cy)]
            self.handle = rendering.make_polygon(vec, filled=True)
            self.handletrans = rendering.Transform()
            self.handle.add_attr(self.handletrans)
            self.handle.set_color(0.1, 0.9, 0.1)

            # create Dama
            self.dama = rendering.make_circle(self.damaradius * scale)
            self.damatrans = rendering.Transform()
            self.dama.add_attr(self.damatrans)
            self.dama.set_color(1, 0, 0)
            # make a hole on Dama
            self.hole = rendering.make_capsule(self.damaradius * 5 / 6 * scale, self.damaradius * 0.4 * scale)
            self.holetrans = rendering.Transform()
            self.hole.add_attr(self.holetrans)
            self.hole.set_color(0.2, 0.2, 0.9)

            # add a string connecting Ken and Dama
            self.string = rendering.Line((0.0, 0.0), (0.0, -self.length * scale))
            self.stringtrans = rendering.Transform()
            self.string.add_attr(self.stringtrans)
            self.string.set_color(1, 1, 1)

            # add geom in displaying order
            self.viewer.add_geom(self.background)
            self.viewer.add_geom(self.dama)
            self.viewer.add_geom(self.hole)
            self.viewer.add_geom(self.ken)
            self.viewer.add_geom(self.handle)
            self.viewer.add_geom(self.string)

        # load the change from self.observation
        ken_y = self.observation[0]
        ken_z = self.observation[1]
        dama_y = self.observation[6]
        dama_z = self.observation[7]
        theta = self.observation[10]
        self.kentrans.set_translation((ken_y - self.y_left) * scale, (ken_z - self.z_bottom) * scale)
        self.handletrans.set_translation((ken_y - self.y_left) * scale, (ken_z - self.z_bottom) * scale)
        self.damatrans.set_translation((dama_y - self.y_left) * scale, (dama_z - self.z_bottom) * scale)
        self.damatrans.set_rotation(theta)
        self.holetrans.set_translation((dama_y - self.y_left) * scale, (dama_z - self.z_bottom) * scale)
        self.holetrans.set_rotation(theta + np.pi / 2)
        if self.tighten:
            self.stringtrans.set_translation((ken_y - self.y_left) * scale, (ken_z - self.z_bottom) * scale)
            self.stringtrans.set_rotation(theta)
        else:
            self.stringtrans.set_translation(-self.length * scale, -self.length * scale)

        self.kentrans.set_rotation(self.beta - np.pi + np.pi / 2)
        self.handletrans.set_rotation(self.beta - np.pi)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
