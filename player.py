'''

Author: Li Shidi
Date: Jun, 2019
Mail address: E0081728@u.nus.edu
National University of Singapore

In this file, the player is allowed to use keyboard to play Kendama

'''

import pygame
import sys
import gym
import time
import numpy as np
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--gain', type=float, default=40.0)
parser.add_argument('--damping', type=float, default=1.0)
args = parser.parse_args()

# gym initialization
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
state = env.reset(init_state=[22.253, 371.043, 0, 0, 0, 0])
env.render()

# keyboard initialization
pygame.init()
scale = 0.5
screen = pygame.display.set_mode((int(scale * env.screen_width),
                                  int(scale * env.screen_height)))
pygame.display.set_caption('pygame event')

# for keyboard control
#command = None
#smooth = 10
# for mouse control
x, y = None, None
gain = args.gain
damping = args.damping
# while True
while True:
    '''
    # read keyboard
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            command = event.key
    pygame.display.update()
    
    # from command to action
    if command == 273:  # up
        action = np.array([0, smooth])
    elif command == 274:  # down
        action = np.array([0, -smooth])
    elif command == 275:  # right
        action = np.array([smooth, 0])
    elif command == 276:  # left
        action = np.array([-smooth, 0])
    else:
        action = np.zeros(2)
    command = None
    '''
    # read mouse motion
    for event in pygame.event.get():
        if event.type == pygame.MOUSEMOTION:
            x, y = event.pos
    pygame.display.update()

    # from command to action
    if x is not None:
        # calculate the current position
        cur_x = (state[0] - env.y_left)
        cur_y = (state[1] - env.z_bottom)
        # calculate the reference position with respect to gym window
        ref_x = x / scale
        ref_y = env.screen_height - y / scale
        # error
        e_x, e_y = ref_x - cur_x, ref_y - cur_y
        # implement a 2nd-order controller
        a_x = gain * e_x - 2 * damping * np.sqrt(gain) * state[2]
        a_y = gain * e_y - 2 * damping * np.sqrt(gain) * state[3]
        action = np.array([a_x, a_y])
    else:
        action = np.zeros(2)

    # run action in gym env
    state, _, done, _ = env.step(action)
    env.render()
    if done:
        time.sleep(1.0)
        state = env.reset(init_state=[22.253, 371.043, 0, 0, 0, 0])
        time.sleep(1.0)
        continue

'''
# the complete pygame event code
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if event.unicode == '':
                print('[key down]', ' #', event.key, event.mod)
            else:
                print('[key down]', ' #', event.unicode, event.key, event.mod)
        elif event.type == pygame.MOUSEMOTION:
            print('[mouse motion]', ' #', event.pos, event.rel, event.buttons)
        elif event.type == pygame.MOUSEBUTTONUP:
            print('[mouse button up]', ' #', event.pos, event.button)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            print('[mouse button down]', ' #', event.pos, event.button)
    pygame.display.update()
'''
