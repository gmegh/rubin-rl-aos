import sys
sys.path.append(r'/home/guillemmh/aos/RL aos/')
from TelescopeEnv import Telescope

sys.path.append(r'/home/guillemmh/aos/RL aos/spinningup/')
from spinup import sac_pytorch as sac
import torch


env_fn = lambda : Telescope('train_test_13')

logger_kwargs = dict(output_dir='/home/guillemmh/aos/RL aos/results/sac_baseline/', exp_name='experiment')

sac(env_fn=env_fn, steps_per_epoch=100, epochs=20, start_steps = 20, max_ep_len=1, update_after=15, update_every=10, logger_kwargs=logger_kwargs)