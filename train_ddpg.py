import sys
sys.path.append(r'/home/guillemmh/aos/RL aos/')
from TelescopeEnv import Telescope

sys.path.append(r'/home/guillemmh/aos/RL aos/spinningup/')
from spinup import ddpg_pytorch as ddpg


env_fn = lambda : Telescope('train_test_11')

logger_kwargs = dict(output_dir='/home/guillemmh/aos/RL aos/results/ddpg_baseline/', exp_name='experment')

ddpg(env_fn = env_fn, seed=0, steps_per_epoch=100, training_steps=10,  epochs=20, replay_size=10000, gamma=0.99, polyak=0.995, pi_lr=0.001, q_lr=0.001, batch_size=100, start_steps=100, update_after=20, update_every=20, act_noise=0.1, num_test_episodes=10, max_ep_len=1, logger_kwargs=logger_kwargs, save_freq=1)