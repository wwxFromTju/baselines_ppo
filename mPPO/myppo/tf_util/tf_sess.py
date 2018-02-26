import tensorflow as tf


def get_session():
    """Returns recently made Tensorflow session"""
    return tf.get_default_session()

def make_session(num_cpu):
    """Returns a session that will use <num_cpu> CPU's only"""
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu)
    return tf.Session(config=tf_config)

def single_threaded_session():
    """Returns a session which will only use a single CPU"""
    return make_session(1)
