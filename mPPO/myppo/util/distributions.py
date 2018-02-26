# import tensorflow as tf
# import numpy as np
# import myppo.tf_util.tf_math as UM
# from tensorflow.python.ops import math_ops
#
# class Pd(object):
#     """
#     A particular probability distribution
#     """
#     def flatparam(self):
#         raise NotImplementedError
#     def mode(self):
#         raise NotImplementedError
#     def neglogp(self, x):
#         # Usually it's easier to define the negative logprob
#         raise NotImplementedError
#     def kl(self, other):
#         raise NotImplementedError
#     def entropy(self):
#         raise NotImplementedError
#     def sample(self):
#         raise NotImplementedError
#     def logp(self, x):
#         return - self.neglogp(x)
#
# class PdType(object):
#     """
#     Parametrized family of probability distributions
#     """
#     def pdclass(self):
#         raise NotImplementedError
#     def pdfromflat(self, flat):
#         return self.pdclass()(flat)
#     def param_shape(self):
#         raise NotImplementedError
#     def sample_shape(self):
#         raise NotImplementedError
#     def sample_dtype(self):
#         raise NotImplementedError
#
#     def param_placeholder(self, prepend_shape, name=None):
#         return tf.placeholder(dtype=tf.float32, shape=prepend_shape+self.param_shape(), name=name)
#     def sample_placeholder(self, prepend_shape, name=None):
#         return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape+self.sample_shape(), name=name)
#
# class CategoricalPdType(PdType):
#     def __init__(self, ncat):
#         self.ncat = ncat
#     def pdclass(self):
#         return CategoricalPd
#     def param_shape(self):
#         return [self.ncat]
#     def sample_shape(self):
#         return []
#     def sample_dtype(self):
#         return tf.int32
#
#
# class MultiCategoricalPdType(PdType):
#     def __init__(self, low, high):
#         self.low = low
#         self.high = high
#         self.ncats = high - low + 1
#     def pdclass(self):
#         return MultiCategoricalPd
#     def pdfromflat(self, flat):
#         return MultiCategoricalPd(self.low, self.high, flat)
#     def param_shape(self):
#         return [sum(self.ncats)]
#     def sample_shape(self):
#         return [len(self.ncats)]
#     def sample_dtype(self):
#         return tf.int32
#
# class DiagGaussianPdType(PdType):
#     def __init__(self, size):
#         self.size = size
#     def pdclass(self):
#         return DiagGaussianPd
#     def param_shape(self):
#         return [2*self.size]
#     def sample_shape(self):
#         return [self.size]
#     def sample_dtype(self):
#         return tf.float32
#
# class BernoulliPdType(PdType):
#     def __init__(self, size):
#         self.size = size
#     def pdclass(self):
#         return BernoulliPd
#     def param_shape(self):
#         return [self.size]
#     def sample_shape(self):
#         return [self.size]
#     def sample_dtype(self):
#         return tf.int32
#
#
# class CategoricalPd(Pd):
#     def __init__(self, logits):
#         self.logits = logits
#     def flatparam(self):
#         return self.logits
#     def mode(self):
#         return UM.argmax(self.logits, axis=-1)
#     def neglogp(self, x):
#         # return tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=x)
#         # Note: we can't use sparse_softmax_cross_entropy_with_logits because
#         #       the implementation does not allow second-order derivatives...
#         one_hot_actions = tf.one_hot(x, self.logits.get_shape().as_list()[-1])
#         return tf.nn.softmax_cross_entropy_with_logits(
#             logits=self.logits,
#             labels=one_hot_actions)
#     def kl(self, other):
#         a0 = self.logits - UM.max(self.logits, axis=-1, keepdims=True)
#         a1 = other.logits - UM.max(other.logits, axis=-1, keepdims=True)
#         ea0 = tf.exp(a0)
#         ea1 = tf.exp(a1)
#         z0 = UM.sum(ea0, axis=-1, keepdims=True)
#         z1 = UM.sum(ea1, axis=-1, keepdims=True)
#         p0 = ea0 / z0
#         return UM.sum(p0 * (a0 - tf.log(z0) - a1 + tf.log(z1)), axis=-1)
#     def entropy(self):
#         a0 = self.logits - UM.max(self.logits, axis=-1, keepdims=True)
#         ea0 = tf.exp(a0)
#         z0 = UM.sum(ea0, axis=-1, keepdims=True)
#         p0 = ea0 / z0
#         return UM.sum(p0 * (tf.log(z0) - a0), axis=-1)
#     def sample(self):
#         u = tf.random_uniform(tf.shape(self.logits))
#         return tf.argmax(self.logits - tf.log(-tf.log(u)), axis=-1)
#     @classmethod
#     def fromflat(cls, flat):
#         return cls(flat)
#
# class MultiCategoricalPd(Pd):
#     def __init__(self, low, high, flat):
#         self.flat = flat
#         self.low = tf.constant(low, dtype=tf.int32)
#         self.categoricals = list(map(CategoricalPd, tf.split(flat, high - low + 1, axis=len(flat.get_shape()) - 1)))
#     def flatparam(self):
#         return self.flat
#     def mode(self):
#         return self.low + tf.cast(tf.stack([p.mode() for p in self.categoricals], axis=-1), tf.int32)
#     def neglogp(self, x):
#         return tf.add_n([p.neglogp(px) for p, px in zip(self.categoricals, tf.unstack(x - self.low, axis=len(x.get_shape()) - 1))])
#     def kl(self, other):
#         return tf.add_n([
#                 p.kl(q) for p, q in zip(self.categoricals, other.categoricals)
#             ])
#     def entropy(self):
#         return tf.add_n([p.entropy() for p in self.categoricals])
#     def sample(self):
#         return self.low + tf.cast(tf.stack([p.sample() for p in self.categoricals], axis=-1), tf.int32)
#     @classmethod
#     def fromflat(cls, flat):
#         raise NotImplementedError
#
# class DiagGaussianPd(Pd):
#     def __init__(self, flat):
#         self.flat = flat
#         mean, logstd = tf.split(axis=len(flat.shape)-1, num_or_size_splits=2, value=flat)
#         self.mean = mean
#         self.logstd = logstd
#         self.std = tf.exp(logstd)
#     def flatparam(self):
#         return self.flat
#     def mode(self):
#         return self.mean
#     def neglogp(self, x):
#         return 0.5 * UM.sum(tf.square((x - self.mean) / self.std), axis=-1) \
#                + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
#                + UM.sum(self.logstd, axis=-1)
#     def kl(self, other):
#         assert isinstance(other, DiagGaussianPd)
#         return UM.sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) - 0.5, axis=-1)
#     def entropy(self):
#         return UM.sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)
#     def sample(self):
#         return self.mean + self.std * tf.random_normal(tf.shape(self.mean))
#     @classmethod
#     def fromflat(cls, flat):
#         return cls(flat)
#
# class BernoulliPd(Pd):
#     def __init__(self, logits):
#         self.logits = logits
#         self.ps = tf.sigmoid(logits)
#     def flatparam(self):
#         return self.logits
#     def mode(self):
#         return tf.round(self.ps)
#     def neglogp(self, x):
#         return UM.sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.to_float(x)), axis=-1)
#     def kl(self, other):
#         return UM.sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=other.logits, labels=self.ps), axis=-1) - UM.sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps), axis=-1)
#     def entropy(self):
#         return UM.sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.ps), axis=-1)
#     def sample(self):
#         u = tf.random_uniform(tf.shape(self.ps))
#         return tf.to_float(math_ops.less(u, self.ps))
#     @classmethod
#     def fromflat(cls, flat):
#         return cls(flat)
#
# def make_pdtype(ac_space):
#     from gym import spaces
#     if isinstance(ac_space, spaces.Box):
#         assert len(ac_space.shape) == 1
#         return DiagGaussianPdType(ac_space.shape[0])
#     elif isinstance(ac_space, spaces.Discrete):
#         return CategoricalPdType(ac_space.n)
#     elif isinstance(ac_space, spaces.MultiDiscrete):
#         return MultiCategoricalPdType(ac_space.low, ac_space.high)
#     elif isinstance(ac_space, spaces.MultiBinary):
#         return BernoulliPdType(ac_space.n)
#     else:
#         raise NotImplementedError
#
# def shape_el(v, i):
#     maybe = v.get_shape()[i]
#     if maybe is not None:
#         return maybe
#     else:
#         return tf.shape(v)[i]
#
#
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import math_ops

import myppo.tf_util.tf_math as UM


class Pd(object):
    """
    A particular probability distribution
    """
    def flatparam(self):
        raise NotImplementedError

    def mode(self):
        raise NotImplementedError

    def neglogp(self, x):
        # Usually it's easier to define the negative logprob
        raise NotImplementedError

    def kl(self, other):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

    def logp(self, x):
        return - self.neglogp(x)


class PdType(object):
    """
    Parametrized family of probability distributions
    """
    def pdclass(self):
        raise NotImplementedError

    def pdfromflat(self, flat):
        return self.pdclass()(flat)

    def param_shape(self):
        raise NotImplementedError

    def sample_shape(self):
        raise NotImplementedError

    def sample_dtype(self):
        raise NotImplementedError

    def param_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=tf.float32, shape=prepend_shape+self.param_shape(), name=name)

    def sample_placeholder(self, prepend_shape, name=None):
        return tf.placeholder(dtype=self.sample_dtype(), shape=prepend_shape+self.sample_shape(), name=name)

class DiagGaussianPdType(PdType):
    def __init__(self, size):
        self.size = size

    def pdclass(self):
        return DiagGaussianPd

    def param_shape(self):
        return [2*self.size]

    def sample_shape(self):
        return [self.size]

    def sample_dtype(self):
        return tf.float32


class DiagGaussianPd(Pd):
    def __init__(self, flat):
        self.flat = flat
        mean, logstd = tf.split(axis=len(flat.shape)-1, num_or_size_splits=2, value=flat)
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def flatparam(self):
        return self.flat

    def mode(self):
        return self.mean

    def neglogp(self, x):
        return 0.5 * UM.sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.to_float(tf.shape(x)[-1]) \
               + UM.sum(self.logstd, axis=-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return UM.sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return UM.sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    @classmethod
    def fromflat(cls, flat):
        return cls(flat)


def make_pdtype(ac_space):
    from gym import spaces
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1
        return DiagGaussianPdType(ac_space.shape[0])
    else:
        raise NotImplementedError


def shape_el(v, i):
    maybe = v.get_shape()[i]
    if maybe is not None:
        return maybe
    else:
        return tf.shape(v)[i]

