import copy

import tensorflow as tf

clip = tf.clip_by_value

def sum(x, axis=None, keepdims=False):
    axis = None if axis is None else [axis]
    return tf.reduce_sum(x, axis=axis, keep_dims=keepdims)

def mean(x, axis=None, keepdims=False):
    axis = None if axis is None else [axis]
    return tf.reduce_mean(x, axis=axis, keep_dims=keepdims)


def max(x, axis=None, keepdims=False):
    axis = None if axis is None else [axis]
    return tf.reduce_max(x, axis=axis, keep_dims=keepdims)

def concatenate(arrs, axis=0):
    return tf.concat(axis=axis, values=arrs)

def argmax(x, axis=None):
    return tf.argmax(x, axis=axis)

def switch(condition, then_expression, else_expression):
    x_shape = copy.copy(then_expression.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'),
                lambda: then_expression,
                lambda: else_expression)
    x.set_shape(x_shape)
    return x