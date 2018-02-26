import myppo.tf_util.tf_type as U
import myppo.tf_util.tf_nn as UN
import tensorflow as tf, numpy as np
from mpi4py import MPI

def traj_show(pi, env, horizon, stochastic):
    ob = env.reset()
    for i in range(horizon):
        ac, vpred = pi.act(stochastic, ob)
        ob, rew, new, _ = env.step(ac)
        env.render()
        if new:
            ob = env.reset()

def test(env, policy_func, *,
        timesteps_per_actorbatch, env_name=None):

    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)

    sess = tf.get_default_session()
    saver = tf.train.Saver()
    saver.restore(sess, './weights/' + env_name + '/' + env_name + '.cptk')
    print('hehe')

    if MPI.COMM_WORLD.Get_rank() == 0:
        for i in range(10):
            print('in')
            traj_show(pi, env, timesteps_per_actorbatch, stochastic=False)


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
