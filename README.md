# 2d Kendama simulator based on OpenAI gym

## How to install
###### $ pip install numpy gym pygame matplotlib

####   Move the folder 'user' to the environment library of python interpreter, for example, ~/.local/lib/python3.7/site-package/gym/envs/.
####   Add the following code to the end of ~/.local/lib/python3.7/site-package/gym/envs/--init--.py
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
###### $ pyhton test_optimal.py
###  For manual control
###### $ python player.py --gain 40.0 --damping 0.8
### For learning process of MBMF learning
###### $ python main.py

