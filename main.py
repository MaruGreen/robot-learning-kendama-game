'''

Author:        Li Shidi
Initialized:   Jun, 2019

This is the simulation of our paper 'Robot Trajectory Learning Framework for Precise Tasks with Discontinuous Dynamics
and High Speed'. In this paper, we propose a practical framework for the robot manipulator to learn the trajectory for
the kendama task, as well as tasks with similar properties. Firstly, the program reads a trajectory from demo/ as the
initial trajectory. The trajectory is learned from human demonstration. Secondly, a simple pendulum dynamic model (user/
kendama_v0.py) is used so as to apply the Differential Dynamic Programming (NaiveDDP.py) to produce a sub-optimal
trajectory in the simulator. Thirdly, the modified Policy Learning by Weighting Exploration with Returns (NaivePoWER.py)
together with Dynamic Movement Primitive (NaiveDMP.py) is applied to this sub-optimal trajectory to conduct the model-
free exploration directly on the physical robot (KUKA LBR iiwa 7 R800).

'''

import numpy as np
import gym
#import matplotlib.pyplot as plt
from NaiveDDP import NaiveDDP
from NaiveDMP import NaiveDMP
from NaivePoWER import NaivePoWER

def readMatrix(fileName):
    # read the information from the file
    file_obj = open(fileName, 'r')
    try:
        all_the_text = file_obj.read()
    finally:
        file_obj.close()
    # conduct the string into np.matrix
    return np.matrix(all_the_text)

def diff(input, dt):
    input = np.matrix(input)
    size = input.shape
    output = np.zeros([size[0]-1, size[1]])
    for i in range(0, size[0]-1):
        output[i] = (input[i+1] - input[i]) / dt
    return output

def dynm_derivative(env, state, action, second_order=False):
    '''
    The dynamics derivative is state-dependent, we should pass the function into the DDP object.
    For different DDP task, the user should define his own dynm_derivative and pass to the object.
    '''
    theta = state[4]
    # only 6 states are independent and 1 for constant term (see paper for more information)
    n = 6 + 1
    m = 2
    fx = np.eye(n)
    fu = np.zeros([n, m])
    if not np.isnan(theta):
        sin = np.sin(theta) * env.friction * env.dt / env.length
        cos = np.cos(theta) * env.friction * env.dt / env.length
        # from matrix A
        fx[0][2] = env.dt
        fx[1][3] = env.dt
        fx[4][5] = env.dt
        fx[5][6] = -env.gravity * sin
        # from x .* dA/dx
        fx[5][4] -= env.gravity * cos
        # from u .* dB/dx
        fx[5][4] += action[0] * sin - action[1] * cos
        # fu = matrix B
        fu[2][0] = env.dt
        fu[3][1] = env.dt
        fu[5][0] = -cos
        fu[5][1] = -sin
    else:
        # now the string is not tightened, robot loses control of theta and omega
        fx[4][4] = 0
        fx[5][5] = 0
        fx[0][2] = env.dt
        fx[1][3] = env.dt
        # fu also does not contain the omega and theta parts
        fu[2][0] = env.dt
        fu[3][1] = env.dt

    # the 2nd-order derivative
    fxx = np.zeros([n, n, n])
    fxu = np.zeros([n, n, m])
    fuu = np.zeros([n, m, m])
    if not second_order:
        return fx, fu, fxx, fxu, fuu
    '''
    Skip the second-order temporarily, will implement later.
    In most of cases, considering only the 1st-order will be enough.
    '''
    ### THE IMPLEMENTATIOM HERE ###
    return fx, fu, fxx, fxu, fuu

def control_limit(input):
    '''
    When DDP is used on physical robot, the required acceleration of the generated trajectories should be constrained.
    See more information at (Tassa, ICRA 2013)
    '''
    limit = 5000    # mm / s2
    for i in range(input.shape[0]):
        if input[i] > limit:
            input[i] = limit
        elif input[i] < -limit:
            input[i] = -limit

def model_based_learning(init_traj, dict, iter_num=40):
    '''
    Create an object to conduct the Differential Dynamic Programming.
    '''
    # load the initial trajectory from human demonstration
    temp = readMatrix(init_traj)
    full_trajectory = np.reshape(temp, [round(temp.shape[1] / 8), 8])
    position = full_trajectory[0:350, [1, 3]] * np.matrix([[-1, 0], [0, 1]])
    velocity = diff(position, 0.01)
    acceleration = diff(velocity, 0.01)

    # create the gym env
    env = gym.make('Kendama-v0').unwrapped
    env.setParameter(dict=dict)
    init_state = np.zeros(6)
    init_state[0:2] = position[0]
    init_state[2:4] = velocity[0]
    # run the env once to obtain a reference trajectory
    obs = env.reset(init_state=init_state)
    ref_traj = np.zeros([349, 13])
    ref_traj[0, :] = obs
    for i in range(348):
        obs, _, _, _ = env.step(acceleration[i, :])
        env.render()
        ref_traj[i + 1, :] = obs

    # define the Q and R matrix for DDP running cost
    Qrun = np.eye(7)
    Rrun = 0.0018 * np.eye(2)
    Qrun[0][0] = 1.0
    Qrun[1][1] = 1.0
    Qrun[2][2] = 0.05
    Qrun[3][3] = 0.05
    Qrun[4][4] = 0.0005
    Qrun[5][5] = 0.001
    Qrun[6][6] = 0.0

    # create the DDP object
    '''
    ref_traj:              349 x 13
    ref_traj[:, 0:6]:      349 x 6   only the first 6 elements are independent, stack the constant 1
    acceleration:          348 x 2
    '''
    obj = NaiveDDP(env, np.hstack((ref_traj[:, 0:6], np.ones([349, 1]))), acceleration,
                   dynm_derivative=dynm_derivative,
                   Qrun=Qrun,
                   Rrun=Rrun,
                   control_limit=control_limit,
                   render=False)
    '''
    Set the desired intermediate points by final_info
    t_b = 2.78s
    t_c = 3.15s
    '''
    final_info = {'num': 2,
                  'keyframe': [278, 315],
                  'desired': [[116, 507, -758, 512, 1.8326, 2.217, 1],
                              [136, 448, 0, 0, 0, 0, 1]],
                  'Q': [np.diag([100, 100, 100, 100, 8000, 8000, 0]),
                        np.diag([280, 280, 0, 0, 0, 0, 0])]}
    for _ in range(iter_num):
        print('DDP iteration:', _)
        obj.update(final_info=final_info)

    return obj

def model_free_imitation(traj_100):
    '''
    Use Dynamic Movement Primitive with constant start and goal points to parameterize the sub-optimal trajectory.
    The weight of the forcing term of DMP can be used as policy parameters for model free Reinforcement Learning.
    '''
    # initialize the time axis
    Freq = 100
    totalTime = 3.5
    time_axis = np.array(np.arange(1 / Freq, totalTime + 1e-10, 1 / Freq))

    # create the kernel
    N = 27
    ti = np.arange(totalTime / 2 / N, totalTime * (1 - 1 / 2 / N) + 1e-10, totalTime / N)
    '''
    # Jan 3, 2019: Change ti in this way, you will see DMP imitate the DDP results much better
    # This is allowed in simulator. However, the generated trajectories will be too jerky for physical robot to execute.
    b = np.arange(2.6, 2.9+1e-10, 0.03)
    ti = np.append(np.append(ti[0:13], b), ti[14:17])
    N = ti.shape[0]
    print('The modified kernel number for one dimension is', N)
    '''
    ci = np.exp(-ti)
    hi = N / np.power(ci, 2)

    # create an object (self, Kgain, Dgain, tau, alpha, t, hi, ci, traj)
    aDMP = NaiveDMP(100, 20, 1, 1, time_axis, hi, ci, traj_100)

    # set the same start and goal points to generate open-loop trajectory
    aDMP.setStartAndGoal(traj_100[:, 0], traj_100[:, -1])

    # Learning from demonstration
    if aDMP.LearningFromDemo():
        '''
        # learning from demonstration
        new = aDMP.GenerationFromLib()
        plt.plot(traj_100[:, :].T, label='demo')
        plt.plot(new[:, :].T, label='new')
        plt.legend()
        plt.show()
        '''
        print("Complete the imitation learning!")

    return aDMP

def model_free_learning(env, dmp, iter_num=150, exploration_scale=2):
    '''
    PoWER.
    '''
    # create the object for reinforcement learning
    beta0 = np.matrix([[exploration_scale, exploration_scale]])
    # NaivePoWER(dmp, iter, bestRoll, bestNoise, beta0, correlated)
    obj = NaivePoWER(dmp, iter_num, 10, 20, beta0, True)

    # the iteration
    for i in range(iter_num + 1):
        print('PoWER iteration:', i)
        position = dmp.GenerationFromLib().T
        velocity = diff(position, 0.01)
        acceleration = diff(velocity, 0.01)
        init_state = np.zeros(6)
        init_state[0:2] = position[0]
        init_state[2:4] = velocity[0]
        env.reset(init_state=init_state)
        t = 0
        final_reward = 0
        while t < 348:
            _, reward, _, _ = env.step(acceleration[t])
            if reward != 0:
                final_reward = reward
            if i % 100 == 0:
                env.render()
            t += 1
        obj.evaluate(i, final_reward)

        if (i + 1) % 100 == 0:
            obj.update(i + 1, mean=True)
        else:
            obj.update(i + 1)

    return obj

if __name__=='__main__':
    # some parameters can be set here
    param = {
        'mass': 1.0,
        'length': 400.0 - 0.0,
        'kenlength': 30.0,
        'kencorner': (25.0, 15.0),
        'damaradius': 35.0,
        'gravity': 9780.0,
        'friction': 0.825,
        'dt': 0.01,
        # theta - phi, dis, theta - ideal_end
        'idealend': 67,
        'gamma': [1, 10, 1],
        'alpha': [5e-3, 1e-4, 5e-3]
    }

    # conduct the model based optimization to the initial trajectory from human demonstration
    obj_ddp = model_based_learning('demo/x105_z122_L400.txt', param, 40)

    '''
    Note that, for simulator, the trajectory performs well already to complete the task. However, if we just let the
    physical robot execute such a trajectory, there will be a high probability that the task fails. The reasons can be 
    model accuracy, mechanism errors, tracking control errors and so on. 
    '''

    # env.step one more time to make trajectory length become 350
    _, _, _, info = obj_ddp.env.step([0, 0])
    sub_optimal = np.vstack((obj_ddp.cur_traj[:, 0:2], info['state'][0:2].reshape([1, 2])))

    # make use of DMP to imitate the resultant trajectory of DDP
    obj_dmp = model_free_imitation(np.matrix(sub_optimal).T)

    '''
    After using DMP to imitate and parameterize the resultant trajectory of DDP, new errors are introduced.
    The trajectory generated by the initial DMP parameters fails in the task!
    '''

    # make use of PoWER, the model free method to generate a smooth trajectory for physical robot to execute
    obj_power = model_free_learning(obj_ddp.env, obj_dmp, 100, 0.1)

    '''
    It takes around 20 rollouts of PoWER learning on the physical robot to complete the task.
    It is possible that it takes infinity rollouts and still can not find the successful trajectory.
    That is because model free Reinforcement Learning sometimes you need to be 'lucky'.
    It is a common evaluation method that you change the random seed and do multiple runs of experiments.
    '''




