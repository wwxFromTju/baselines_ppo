import numpy as np
import tensorflow as tf  # pylint: ignore-module
import builtins
import functools
import copy
import os
import collections

VARIABLES = {}

_PLACEHOLDER_CACHE = {}  # name -> (placeholder, dtype, shape)

def is_placeholder(x):
    return type(x) is tf.Tensor and len(x.op.inputs) == 0

# class TfInput(object):
#     def __init__(self, name="(unnamed)"):
#         """Generalized Tensorflow placeholder. The main differences are:
#             - possibly uses multiple placeholders internally and returns multiple values
#             - can apply light postprocessing to the value feed to placeholder.
#         """
#         self.name = name
#
#     def get(self):
#         """Return the tf variable(s) representing the possibly postprocessed value
#         of placeholder(s).
#         """
#         raise NotImplemented()
#
#     def make_feed_dict(data):
#         """Given data input it to the placeholder(s)."""
#         raise NotImplemented()


def get_placeholder(name, dtype, shape):
    if name in _PLACEHOLDER_CACHE:
        out, dtype1, shape1 = _PLACEHOLDER_CACHE[name]
        assert dtype1 == dtype and shape1 == shape
        return out
    else:
        out = tf.placeholder(dtype=dtype, shape=shape, name=name)
        _PLACEHOLDER_CACHE[name] = (out, dtype, shape)
        return out

def get_placeholder_cached(name):
    return _PLACEHOLDER_CACHE[name][0]

def reset():
    global _PLACEHOLDER_CACHE
    global VARIABLES
    _PLACEHOLDER_CACHE = {}
    VARIABLES = {}
    tf.reset_default_graph()
