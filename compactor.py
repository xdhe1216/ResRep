# -*- coding: utf-8 -*-
"""CompactorLayer layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.layers import InputSpec
from keras.utils import conv_utils
import keras.backend as K
import tensorflow as tf
import keras
import warnings
import numpy as np
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Layer


class CompactorLayer(Layer):
    def __init__(self, filters,
                 rank=2,
                 kernel_size=1,
                 strides=1,
                 padding='same',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=False,
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(CompactorLayer, self).__init__(**kwargs)
        self.rank = rank
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)
        self.filters = filters
        self.kernel = None
        self.mask = None
        self.bias = None

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=initializers.constant(np.eye(self.filters)),
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      )
        self.mask = self.add_weight(shape=(1, 1, 1, self.filters),
                                    initializer=tf.ones_initializer(),
                                    name='mask',
                                    trainable=False
                                    )
        self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, **kwargs):
        temp_kernel = self.kernel * self.mask
        if self.rank == 2:
            outputs = K.conv2d(
                inputs,
                temp_kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer': regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint)
        }
        base_config = super(CompactorLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CompactorMonitor(keras.callbacks.Callback):
    def __init__(self, monitor='val_loss', min_delta=0, min_flops=0.5, patience=5, step=2, verbose=1, mode='auto'):
        super(CompactorMonitor, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.min_delta = min_delta
        self.min_flops = min_flops
        self.wait = 0
        self.step = step
        self.filter_num = 0
        self.best = np.Inf
        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self, logs=None):
        self.wait = 0
        for layer in self.model.layers:
            if 'compactor_layer' in layer.name:
                w, m = self.model.get_layer(layer.name).get_weights()
                self.filter_num += np.sum(m == 0)
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        current_flops, orignal_flops = self.get_flops()
        print("Decline FLOPs Percentage:", 100*(1 - current_flops / orignal_flops))

    def get_flops(self):
        orignal_total_para_nums = 0
        orignal_total_flops = 0
        current_total_para_nums = 0
        current_total_flops = 0
        layer_num = len(self.model.layers)
        for i, layer in enumerate(self.model.layers):
            if 'conv' not in layer.name:
                continue
            kernel_size = layer.kernel_size
            original_input_size = list(layer.input_shape)
            original_output_size = list(layer.output_shape)
            current_input_size = original_input_size.copy()
            current_output_size = original_output_size.copy()

            for j in range(1, layer_num-i):
                next_layer = self.model.layers[min(i + j, layer_num - 1)]
                if 'compactor' in next_layer.name:
                    current_output_size[-1] = np.sum(next_layer.get_weights()[1] == 1)
                    break

            for j in range(i):
                pre_layer = self.model.layers[max(i - j, 0)]
                if 'compactor' in pre_layer.name:
                    current_input_size[-1] = np.sum(pre_layer.get_weights()[1] == 1)
                    break

            temp_kernel = kernel_size[0] * kernel_size[1]
            if layer.bias is None:
                temp = temp_kernel * original_input_size[-1]
                orignal_para_nums = temp * original_output_size[-1]
                orignal_flops = (2 * temp - 1) * original_output_size[-1] \
                                * original_output_size[-2] * original_output_size[-3]

                temp = temp_kernel * current_input_size[-1]
                current_para_nums = temp * current_output_size[-1]
                current_flops = (2 * temp - 1) * current_output_size[-1] * current_output_size[-2] * current_output_size[-3]

            else:
                temp = temp_kernel * original_input_size[-1]
                orignal_para_nums = (temp + 1) * original_output_size[-1]
                orignal_flops = 2 * temp * original_output_size[-1] * original_output_size[-2] * original_output_size[
                    -3]

                temp = temp_kernel * current_input_size[-1]
                current_para_nums = (temp + 1) * current_output_size[-1]
                current_flops = 2 * temp * current_output_size[-1] * current_output_size[-2] * current_output_size[-3]
            orignal_total_para_nums += orignal_para_nums
            orignal_total_flops += orignal_flops
            current_total_para_nums += current_para_nums
            current_total_flops += current_flops
        return current_total_flops, orignal_total_flops

    def on_epoch_end(self, epoch, logs=None):
        # 1) get weights
        w_dict = {}
        for layer in self.model.layers:
            if 'compactor_layer' in layer.name:
                w, m = self.model.get_layer(layer.name).get_weights()
                temp_w = np.sqrt(np.power(w, 2).sum((0, 1, 2)))
                for i in range(temp_w.shape[0]):
                    w_dict[layer.name + '-' + str(i)] = temp_w[i]
        w_dict = sorted(w_dict.items(), key=lambda kv: (kv[1], kv[0]))

        # 2) stop rules
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
            return
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.filter_num += self.step
                self.wait = 0
                current_flops, orignal_flops = self.get_flops()
                decline_flops = 1 - current_flops / orignal_flops
                print("Decline FLOPs Percentage:", decline_flops * 100)
                if decline_flops > self.min_flops:
                    self.model.stop_training = True
                    w_dict = {}

        # 3) set mask value
        for num, [n, _] in enumerate(w_dict):
            name, idex = n.split('-')
            w, m = self.model.get_layer(name).get_weights()

            if num >= self.filter_num:
                m[:, :, :, int(idex)] = 1
            else:
                remain_num = np.sum(m == 1)
                if remain_num == 1:
                    # print(name)
                    continue
                m[:, :, :, int(idex)] = 0
            self.model.get_layer(name).set_weights([w, m])

        # 4) show mask information
        if self.verbose > 0:
            for layer in self.model.layers:
                if 'compactor_layer' in layer.name:
                    w, m = self.model.get_layer(layer.name).get_weights()
                    print(layer.name, m)


class Lasso(keras.regularizers.Regularizer):
    def __init__(self, l1=0.):
        self.l1 = K.cast_to_floatx(l1)

    def __call__(self, x):
        regularization = 0.
        regularization += K.sum(self.l1 * x / K.sqrt(K.sum(K.square(x), axis=[0, 1, 2], keepdims=True)))

        return regularization

    def get_config(self):
        return {
            'l1': float(self.l1),
        }


def lasso(learn=0.01):
    return Lasso(l1=learn)


get_custom_objects().update({'Lasso': lasso()})
get_custom_objects().update({'CompactorLayer': CompactorLayer})
