'''

Author:        Li Shidi
Initialized:   Dec, 2017
Revised:       Jun, 2019

This is the module of PoWER algorithm for implementation in kendama simulator.

'''

import numpy as np
from numpy import random as nr
#import matplotlib.pyplot as plt
import time
import os

from NaiveDMP import NaiveDMP

def readMatrix(fileName):
    # read the information from the file
    file_obj = open(fileName, 'r')
    try:
        all_the_text = file_obj.read()
    finally:
        file_obj.close()
    # conduct the string into np.matrix
    return np.matrix(all_the_text)

class NaivePoWER:

    def __init__(self, dmp, iteration, bestRollout, bestNoise, beta0, correlated):

        self.dmp = dmp
        self.start = dmp.traj[:, 0]
        self.goal = dmp.traj[:, -1]
        self.current_traj = np.zeros(dmp.traj.T.shape)
        self.policy_mean = dmp.weight
        self.alpha = 1e-4 # the distance for ball-in-cup is counted in mm
        # define the beta_k here
        k = np.arange(1, iteration + 1.001)
        self.betak = -np.tanh(4 * (k - iteration / 2) / iteration) / 2 + 0.5

        ker_num = self.dmp.kernelNum
        dimension = self.dmp.dimension
        point = self.dmp.pointNum
        print("NaivePoWER: dimension = ", dimension)

        self.iteration = iteration
        self.history_weight = np.zeros([iteration + 1, ker_num, dimension])
        self.history_lambda = np.zeros([iteration + 1, ker_num, dimension])
        self.history_q = np.zeros([point, iteration+1])
        self.importance_table = np.zeros([2, iteration+1])

        self.bestRollout = bestRollout
        self.bestNoise = bestNoise
        self.correlated = correlated
        # do not set beta0 = 0 in simulation, it will not converge
        if np.matrix(beta0).all() == 0:
            self.dmp.LearningFromDemo()
            print(self.dmp.weight)
            print(np.std(self.dmp.weight, 0))
            # use the std as the initial Lambda
            self.beta0 = np.std(self.dmp.weight, 0)
            print(self.beta0)
        else:
            self.beta0 = beta0

        if correlated:
            [temp, self.psi] = np.linalg.eig(self.createPsi(ker_num))
            self.history_lambda[0] = temp.reshape(ker_num, 1).dot(self.beta0).reshape(ker_num, dimension)
        else:
            self.psi = np.eye(ker_num)
            self.history_lambda[0] = np.ones([ker_num, 1]).dot(self.beta0)
        
        # for the real robot case, the initial policy is learned from human demonstration
        self.history_weight[0] = self.dmp.weight

    def createPsi(self, order):
        # create A
        A = np.zeros([order+2, order])
        A[1:-1, :] = -2 * np.eye(order)
        for i in range(0, order):
            A[i, i] = 1
            A[i+2, i] = 1
        # create R
        R = np.matrix(A.T.dot(A))
        return R.I

    def explore(self, lamb):
        # extract the parameters
        ker_num = self.dmp.kernelNum
        dimension = self.dmp.dimension
        lamb = np.matrix(lamb).reshape([ker_num, dimension])

        # create the noise
        noise_bar = np.zeros([ker_num, dimension])
        for j in range(0, dimension):
            for i in range(0, ker_num):
                noise_bar[i, j] = nr.normal(0, np.sqrt(lamb[i, j]))

        noise = self.psi.dot(noise_bar)

        # add the noise to the policy weight
        #plt.plot(noise)
        #plt.axis([-1, self.dmp.kernelNum, -3, 3])
        #plt.show()
        self.dmp.weight = self.dmp.weight + noise

        # return the weight with the noise
        return self.dmp.weight

    def evaluate(self, iter, reward):
        # extract the parameters
        point = self.dmp.pointNum
        # calculate the Q value
        '''
        Jun 9, 2019: In this case, simply set all Q value to the final reward.
        '''
        q_matrix = np.zeros([point, 1])
        for i in range(point):
            q_matrix[i] = reward

        # save the q value in the table
        self.history_q[:, iter] = q_matrix.reshape(point)
        self.importance_table[:, iter] = np.array([iter, self.history_q[0, iter]]).reshape(2)

    def update(self, iter, mean=False):
        # Jan 29, 2018: add the input, mean. If mean == True, do not add noise to the mean policy
        # extract the parameters
        ker_num = self.dmp.kernelNum
        dimension = self.dmp.dimension
        point = self.dmp.pointNum

        # order the importance table
        importance = np.lexsort(-self.importance_table)  # better in front, worse at the back

        # Jan 19, 2018: if we learn without enough good rollouts, the policy seems not going forward
        # This is the real robot, therefore we should only wait until we have enough roullouts
        # which are better than the initial one
        if self.history_q[0, importance[min(iter - 1, self.bestNoise)]] < self.history_q[0, 0]:
            for good in range(0, iter):
                if self.history_q[0, importance[good]] < self.history_q[0, 0]:
                    break
            print("PoWER: Good rollouts not enough: ", good - 1)
        else:
            good = iter
            
        # update the weight for each dimension separately
        for d in range(0, dimension):
            # initialize the weight
            current_nom = np.zeros([ker_num, 1])
            current_dnom = np.zeros([ker_num, ker_num])

            # calculate the update of the policy mean
            for i in range(0, min(good, self.bestRollout)):
                # obtain the rollout number with importance
                j = importance[i]
                # calculate the time-variant matrix W(t)
                temp_W = np.zeros([point, ker_num, ker_num])
                for ii in range(0, point):
                    base = self.dmp.phi_mat.T[:, ii]
                    Sigma = np.diag(self.history_lambda[j, :, d])
                    if self.correlated:
                        Sigma = self.psi.dot(Sigma).dot(self.psi.T)
                    temp_W[ii, :, :] = base.dot(base.T) / (base.T.dot(Sigma).dot(base)+1e-50)
                # calculate the policy exploration with respect to the current one
                temp_explore = self.history_weight[j, :, d].reshape(ker_num) - self.policy_mean[:, d].reshape(ker_num)
                # accumulate the value with q
                for ii in range(0, point):
                    tem = np.matrix(temp_explore).reshape(ker_num, 1)
                    current_nom = current_nom + temp_W[ii, :, :].dot(tem) * self.history_q[ii, j]
                    current_dnom = current_dnom + temp_W[ii, :, :] * self.history_q[ii, j]
            # update
            current_dnom = np.matrix(current_dnom)
            temp_update = current_dnom.I.dot(current_nom)
            self.history_weight[iter, :, d] = self.policy_mean[:, d].reshape(ker_num) + temp_update.reshape(ker_num)

        # calculate the exploration rate
        if self.bestNoise == 0: # not going to learn Lambda
            # Feb 12, 2018: To read the current beta_k from a txt
            if os.path.exists(pathNameCord + "betak.txt"):
                value = readMatrix(pathNameCord + "betak.txt")
                self.betak[iter] = value[0, 0]
            # Feb 3, 2018: function to realize beta_k
            self.history_lambda[iter] = (self.betak[iter]/self.betak[iter - 1]) * self.history_lambda[iter - 1]
            print("PoWER: The rate of beta_k = ", self.betak[iter])
        # want to learn Lambda but not enough good rollouts
        elif good < self.bestRollout:
            # use constant one
            self.history_lambda[iter] = self.history_lambda[iter - 1]
        # learn the exploration rate adaptively
        else:
            # update the variance for all dimensions at one time
            var_nom = np.zeros([ker_num, dimension])
            var_dnom = 0
            for i in range(0, min(good, self.bestNoise)):
                # obtain the rollout number with importance
                j = importance[i]
                # calculate the policy exploration with respect to the updated policy mean
                # that is because in importance sampling, q(x) = p(x)f(x),
                # we should calculate the deviation to the updated mean for the Gaussian distribution
                # this is different from the Jen's code
                temp_explore = self.history_weight[j].reshape([ker_num, dimension]) - self.history_weight[iter]
                if self.correlated:
                    temp_explore = self.psi.T.dot(temp_explore)
                # accumulate the value with q
                var_nom = var_nom + np.power(temp_explore, 2) * sum(self.history_q[:, j])
                var_dnom = var_dnom + sum(self.history_q[:, j])
            # update!
            temp = var_nom / (var_dnom+1e-20)
            temp = np.maximum(temp, 0.1 * self.history_lambda[0])
            temp = np.minimum(temp, 300 * self.history_lambda[0])
            self.history_lambda[iter] = temp.reshape(ker_num, dimension)

        # now we can save the new policy mean to self.policy_mean
        self.policy_mean = self.history_weight[iter]
        self.dmp.weight = self.history_weight[iter]
        # if it is not the last iteration, we add new noise to the updated policy
        # Jan 29, 2018: add mean to see the mean policy during the learning
        if iter != self.iteration and (not mean):
            self.history_weight[iter] = self.explore(self.history_lambda[iter]).reshape(ker_num, dimension)
        else:
            print("PoWER::update: We are using the mean policy!")

        return True

    def save(self, pathName):
        # get the row for saving history_weight and history_lambda
        row = (self.iteration + 1) * self.dmp.kernelNum
        # save the data of this run of learning
        np.savetxt(pathName+"historty_weight.txt", self.history_weight.reshape(row, self.dmp.dimension), fmt='%8e', delimiter='  ', newline='\n')
        np.savetxt(pathName+"history_lambda.txt", self.history_lambda.reshape(row, self.dmp.dimension), fmt='%8e', delimiter='  ', newline='\n')
        np.savetxt(pathName+"history_q.txt", self.history_q, fmt='%8e', delimiter='  ', newline='\n')
        np.savetxt(pathName+"importance_table.txt", self.importance_table, fmt='%8e', delimiter='  ', newline='\n')
