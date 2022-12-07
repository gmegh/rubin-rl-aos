import sys
sys.path.append(r'/home/guillemmh/aos/RL aos/')
from TelescopeEnv import Telescope

sys.path.append(r'/home/guillemmh/aos/RL aos/spinningup/')
from spinup import ppo_pytorch as ppo
import torch


env_fn = lambda : Telescope('train_test_10')

logger_kwargs = dict(output_dir='/home/guillemmh/aos/RL aos/ppo_3/', exp_name='ppo_0')

ppo(env_fn = env_fn, ac_kwargs=dict(), seed=0, steps_per_epoch=100, epochs=20, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3, train_pi_iters=50, train_v_iters=50, lam=0.97, max_ep_len=1,
        target_kl=0.01, logger_kwargs=logger_kwargs, save_freq=10)