'''

Author:        Li Shidi
Initialized:   Dec, 2017
Revised:       Jun, 2019

This is the module of DMP algorithm for implementation in kendama simulator

'''

import numpy as np
#import matplotlib.pyplot as plt

class NaiveDMP:

    def __init__(self, Kgain, Dgain, tau, alpha, t, hi, ci, traj):
        self.Kgain = Kgain
        self.Dgain = Dgain
        self.tau = tau
        self.alpha = alpha
        self.t = np.matrix(t)
        self.hi = np.matrix(hi)
        self.ci = np.matrix(ci)
        self.traj = np.matrix(traj)
        # Jun 8, 2019: add this to use DMP in new kendama simulator
        self.start = None
        self.goal = None

        self.dt = t[1] - t[0]        # time between two time points
        self.kernelNum = hi.size               # the kernel number for each dimension
        self.dimension = self.traj.shape[1-1]  # the number of dimensions
        self.pointNum = self.traj.shape[2-1]   # the number of time points

        self.weight = np.zeros([self.kernelNum, self.dimension])  # initial the weights with zeros
        # initial the activate matrix (regularized)
        activate = self.__phi(self.t.T, self.tau, self.alpha, self.ci, self.hi)
        # Jan 3, 2019: comment this command and cancel the regularization, because I feel it would be better for DMP to coop with DDP
        #self.phi_mat = activate / np.average(activate, axis=1).dot(np.ones(self.ci.shape))
        self.phi_mat = activate
        #plt.plot(self.phi_mat)
        #plt.show()

        # create a differential matrix
        self.diff = -np.eye(self.pointNum)
        self.diff[self.pointNum - 1, self.pointNum - 1] = 0
        for j in range(2, self.pointNum + 1):
            self.diff[j - 1][j - 2] = 1

    def __differetial(self, matrix):
        # conduct the differential
        return matrix.dot(self.diff)

    def __phi(self, time, tau, alpha, c, h):
        if c.shape != h.shape:
            print("Error (NaiveDMP.__phi): The sizes of ci and hi do not fit!")
            return 0
        else:
            phase = self.__phase(time, tau, alpha).dot(np.ones(c.shape))
            h = np.ones(time.shape).dot(h)
            c = np.ones(time.shape).dot(c)
            temp = -np.multiply(np.multiply(h, phase - c), phase - c)
            return np.exp(temp)

    def __phase(self, time, tau, alpha):
        return np.exp(-alpha * time / tau)

    def setStartAndGoal(self, start, goal):
        # Jun 8, 2019: add this function to apply DMP in the new kendama simulator
        self.start = start
        self.goal = goal

    def LearningFromDemo(self):
        # extract some parameters for short
        point = self.pointNum
        K = self.Kgain
        D = self.Dgain
        phi_mat = self.phi_mat

        # conduct the differentiating
        velocity = self.__differetial(self.traj) / self.dt
        velocity[:, point - 1] = 2 * velocity[:, point - 2] - velocity[:, point - 3]
        acceleration = self.__differetial(velocity) / self.dt
        acceleration[:, point - 1] = 2 * acceleration[:, point - 2] - acceleration[:, point - 3]

        # calculate the f_target
        goal = np.ones([point, 1]).dot(self.traj[:, point - 1].T)
        start = np.ones([point, 1]).dot(self.traj[:, 0].T)
        f_target = (-K*(goal - self.traj.T) + D*velocity.T + self.tau*acceleration.T) / (goal - start)

        # check the rank
        temp = phi_mat.T.dot(phi_mat)
        '''
        if np.linalg.matrix_rank(temp) < self.kernelNum:
            print("Warming (NaiveDMP.LearningFromDemo): The kernel number is too large! Use the S method.")
            temp = self.__phase(self.t, self.tau, self.alpha)
            phase = np.ones([self.kernelNum, 1]).dot(temp)
            nom = np.multiply(phi_mat.T, phase).dot(f_target)
            dnom = np.multiply(phi_mat.T, phase).dot(temp.T).dot(np.ones([1, self.dimension]))
            self.weight = nom / dnom
        else:
            self.weight = temp.I.dot(phi_mat.T).dot(f_target)
        '''
        self.weight = temp.I.dot(phi_mat.T).dot(f_target)

        # return True meaning the DMP is learned successfully
        return True

    def GenerationFromLib(self):
        # Jun 8, 2019: add this to use DMP in kendama-v0 simulator
        if self.start is None or self.goal is None:
            print('NaiveDMP: Please call serStartAndGoal first!')
            return -1
        start = self.start
        goal = self.goal

        # extract some parameters for short
        point = self.pointNum
        K = self.Kgain
        D = self.Dgain

        # calculate the f(s)
        fs = self.phi_mat.dot(self.weight)    # it is actually f(t)

        # initialize the trajectory
        new_traj = np.matrix(np.zeros([self.dimension, point]))
        if goal.shape[0] == 1:
            goal = goal.T
        if start.shape[0] == 1:
            start = start.T

        # calculate the new trajectory
        vt = np.matrix(np.zeros(goal.shape))   # t = 1
        xt = start                # t = 1
        for i in range(0, point):
            new_traj[:, i] = xt
            temp = vt
            vt = vt + (K * (goal - xt) - D * vt + np.multiply((goal - start), fs[i, :].T)) * self.dt / self.tau
            xt = xt + temp * self.dt / self.tau
            
        # return the trajectory generated by this DMP
        return new_traj
