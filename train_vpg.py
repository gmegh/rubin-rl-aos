import sys
sys.path.append(r'/home/guillemmh/aos/RL aos/')
from TelescopeEnv import Telescope

sys.path.append(r'/home/guillemmh/aos/RL aos/spinningup/')
from spinup import vpg_pytorch as vpg
import torch


env_fn = lambda : Telescope('train_test_14')

logger_kwargs = dict(output_dir='/home/guillemmh/aos/RL aos/results/vpg_allactions_2/', exp_name='experiment')

vpg(env_fn = env_fn, seed=0, steps_per_epoch=100, epochs=20, gamma=0.99, pi_lr=0.0003, vf_lr=0.001, train_v_iters=50, lam=0.97, max_ep_len=1, logger_kwargs={}, save_freq=10)