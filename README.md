# 2d Kendama simulator based on OpenAI gym

## How to install
###### $ pip install gym
###### $ pip install pygame
###### $ cp user $(YOUR_PATH_TO_gym_envs)/
####   Take my desktop as an example, I run $ sudo cp user ~/.local/lib/python3.7/site-package/gym/envs/
###  Add the following code to the end of $(YOUR_PATH_TO_gym_envs)/__init__.py
###### #### User
###### #### ---------

###### register(
######    id='Kendama-v0',
######    entry_point='gym.envs.user:KendamaEnv0',
######    max_episode_steps=500,
######    reward_threshold=100.0,
######    )


## How to run
###  For automatical control
###### $ pyhton test_model.py
###  For human control
###### $ python player.py


