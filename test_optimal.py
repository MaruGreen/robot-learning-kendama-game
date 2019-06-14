
import gym
import time
import numpy as np

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

if __name__=='__main__':
    env = gym.make('Kendama-v0').unwrapped

    dict = {
        'mass': 1.0,
        'length': 400.0 - 0.0,
        'kenlength': 30.0,
        'kencorner': (25.0, 15.0),
        'damaradius': 35.0,
        'gravity': 9780.0,
        'friction': 0.825,
        'dt': 0.01,
        # theta - phi, dis, theta - idealend
        'idealend': 67,
        'gamma': [1, 1, 1],
        'alpha': [5e-3, 1e-4, 5e-3]
    }
    env.setParameter(dict=dict)

    # from the txt file, get the pre-defined actions
    temp = readMatrix('demo/optimal.txt')
    full_trajectory = np.reshape(temp, [round(temp.shape[1] / 8), 8])
    position = full_trajectory[0:350, [1, 3]] * np.matrix([[-1, 0], [0, 1]])
    velocity = diff(position, 0.01)
    acceleration = diff(velocity, 0.01)

    init_state = np.zeros(6)
    init_state[0:2] = position[0]
    init_state[2:4] = velocity[0]
    obs = env.reset(init_state=init_state)

    # obtain a reference trajectory to calculate the immediate reward
    t = 0
    while 1:
        _, _, _, _ = env.step(acceleration[t])
        env.render()
        # if t >= 280:
        #    time.sleep(0.1)
        t = t + 1
        if t == 348:
            break
