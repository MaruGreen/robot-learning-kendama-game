'''

Author:        Li Shidi
Initialized:   Jun, 2019

This module uses Differential Dynamical Programming as a Model-based RL to optimize open-loop kendama trajectory.
The DDP considers value function as a 2nd-order quadratic model while policy function as a 1st-order linear model.
We define the state-action loss function Q(x,u) = l(x,u) + V(x') = l(x,u) + V(f(x,u))
The core of this algorithm is to select the optimal control input (action) to minimize Q(x,u) at each time step.
To differential Q(x,u) with respect to u, 2nd-order derivative of dynamics f will be required.
It can be seen that in most of the cases, considering only 1st-order derivative of dynamics should be enough.
In this module, we consider only the 1st-order derivative of dynamics.

'''

import numpy as np
import gym
#import matplotlib.pyplot as plt

class NaiveDDP:
    def __init__(self, env, ref_traj, ref_ctrl, dynm_derivative, Qrun, Rrun,
                 dynm_second=False, control_limit=None, render=False):
        self.env = env
        self.dynm_derivative = dynm_derivative
        self.second = dynm_second
        self.control_limit = control_limit
        self.render = render
        self.running_Q = 0.5 * (Qrun + Qrun.T)
        self.running_R = 0.5 * (Rrun + Rrun.T)
        self.cur_ctrl = self.ref_ctrl = ref_ctrl
        self.cur_traj = self.ref_traj = ref_traj

        self.max_step, self.action_dim = ref_ctrl.shape
        self.observation_dim = ref_traj.shape[1]

        self.k = np.zeros([self.max_step, self.action_dim])
        self.K = np.zeros([self.max_step, self.action_dim, self.observation_dim])

        self.update_num = 0

    def update(self, init_state=None, alpha=0.1, final_info=None):
        '''
        init_state: This argument is to be passed to the backward pass.
                    In some DDP application, we can change the initial state and make the robot find a trajectory to
                    some desired goal state. For example, NUS CS6244 UV Navigation project.
        info: This argument is to be passed to the forward pass.
              In some DDP application, we want the robot to go through some particular intermediate points.
              For example, the kendama trajectory planning.
        '''
        # the learning rate for open-loop term ki
        self.alpha = alpha

        # obtain the open-loop term k and the feedback gain K
        self._backward_pass(final_info)

        # run a forward pass and save the current trajectory and control input
        self._forward_pass(init_state)
        self.update_num += 1

    def _backward_pass(self, final_info):
        # initialize the derivative of value for the last time step
        v_x = np.zeros([self.observation_dim, 1])
        v_xx = np.zeros(([self.observation_dim, self.observation_dim]))

        # through the current trajectory
        for i in range(self.max_step, 0, -1):
            state = self.cur_traj[i - 1]
            input = self.cur_ctrl[i - 1]

            # calculate the derivative of dynamics
            fx, fu, fxx, fxu, fuu = self.dynm_derivative(self.env, state, input, self.second)

            # there may be nan in the states, have to add this code to deal with it
            temp = state - self.ref_traj[i - 1]
            n = 0
            for ele in temp:
                if np.isnan(ele):
                    temp[n] = 0
                n += 1
            # calculate the delta Q matrices with respect to running cost
            q_x = fx.T.dot(v_x) + self.running_Q.dot(temp).reshape([self.observation_dim, 1])
            q_u = fu.T.dot(v_x) + self.running_R.dot(input - self.ref_ctrl[i - 1]).reshape([self.action_dim, 1])
            q_xx = self.running_Q + fx.T.dot(v_xx).dot(fx)
            q_uu = self.running_R + fu.T.dot(v_xx).dot(fu)
            q_ux = fu.T.dot(v_xx).dot(fx)
            if self.second:
                fux = np.transpose(fxu, [0, 2, 1])
                q_xx += self._inner_prod(v_x, fxx)
                q_uu += self._inner_prod(v_x, fux)
                q_ux += self._inner_prod(v_x, fuu)

            # consider the final cost (when t = tb or t = tc)
            if final_info is not None:
                for num in range(final_info['num']):
                    if i + 1 == final_info['keyframe'][num]:
                        temp = state - final_info['desired'][num]
                        n = 0
                        for ele in temp:
                            if np.isnan(ele):
                                temp[n] = 0
                            n += 1
                        q_x += final_info['Q'][num].dot(temp).reshape([self.observation_dim, 1])
                        q_xx += final_info['Q'][num]

            # obtain the open-loop term and feedback term, save in sequences
            ki = -np.matrix(q_uu).I.dot(q_u)
            self.k[i - 1, :] = ki.reshape(self.action_dim)
            Ki = -np.matrix(q_uu).I.dot(q_ux)
            self.K[i - 1, :, :] = Ki

            # calculate the derivative of value for the previous time step
            v_x = q_x + Ki.T.dot(q_uu).dot(ki) + Ki.T.dot(q_u) + q_ux.T.dot(ki)
            v_xx = q_xx + Ki.T.dot(q_uu).dot(Ki) + Ki.T.dot(q_ux) + q_ux.T.dot(Ki)
            v_xx = 0.5 * (v_xx + v_xx.T)

    def _forward_pass(self, init_state):
        '''
        This function runs the env model to get a complete trajectory.
        The only argument is the initial state. Together with the self.k and self.K, a trajectory can be received.
        '''
        if init_state is None:
            init_state = self.cur_traj[0]
        self.env.reset(init_state)

        # calculate the trajectory
        next_state = init_state
        for i in range(self.max_step):
            temp = self.cur_traj[i].copy()
            self.cur_traj[i] = next_state
            # there may be nan in states
            margin = self.cur_traj[i] - temp
            n = 0
            for ele in margin:
                if np.isnan(ele):
                    margin[n] = 0
                n += 1
            self.cur_ctrl[i] += self.alpha * self.k[i] + self.K[i].dot(margin)
            # constrain the scale of control input
            if self.control_limit is not None:
                self.control_limit(self.cur_ctrl[i])
            # run the control input for one step
            _, _, _, info = self.env.step(self.cur_ctrl[i])
            if self.render and self.update_num % 10 == 0:
                self.env.render()
            # in DDP, we want the state, not the observation
            next_state = info['state']
        self.cur_traj[-1] = next_state

    def _inner_prod(self, vector, tensor):
        '''
        This function calculate the inner product of a vector and a tensor.
        This will be required we are considering the 2nd-order of dynamics derivative.
        '''
        return np.dot(vector, np.transpose(tensor, [1, 2, 0]))
