#!/usr/bin/env python3

from mpi4py import MPI
import gym, logging

from myppo.util import set_global_seeds
import myppo.tf_util.tf_sess as US
from myppo.ppo import mlp_policy, ppo_policy_test

def train(env_id, num_timesteps, seed):

    rank = MPI.COMM_WORLD.Get_rank()
    sess = US.single_threaded_session()
    sess.__enter__()
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)

    # U.make_session(num_cpu=1).__enter__()
    # set_global_seeds(seed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=3)
    env.seed(seed)

    test_env = gym.make(env_id)
    test_env.seed(seed)
    gym.logger.setLevel(logging.WARN)
    ppo_policy_test.test(env, policy_fn,
                         timesteps_per_actorbatch=2048,
                         env_name=env_id
                         )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(2e7))
    args = parser.parse_args()
    # logger.configure()
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)


if __name__ == '__main__':
    main()
