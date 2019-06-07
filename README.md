## 2d Kendama simulator based on OpenAI gym

###### $ pip install gym
###### $ pip install pygame
###### $ sudo cp user $(YOUR_PATH_TO_gym_envs)/
#####  Add register to $(YOUR_PATH_TO_gym_envs)/__init__.py
####  For automatical control
###### $ pyhton test_model.py
####  For human control
###### $ python player.py

###### #### User
###### #### ---------

###### register(
######    id='Kendama-v0',
######    entry_point='gym.envs.user:KendamaEnv0',
######    max_episode_steps=500,
######    reward_threshold=100.0,
######    )


