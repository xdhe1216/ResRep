from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Activation, BatchNormalization
from keras.models import Sequential
from keras import layers
from keras.models import load_model
from tensorflow.examples.tutorials.mnist import input_data
import keras
import keras.backend as K
import numpy as np
import tensorflow as tf
from compactor import CompactorLayer, lasso, CompactorMonitor
from kerassurgeon.operations import delete_layer, delete_channels

tf_config = tf.compat.v1.ConfigProto()
tf_config.gpu_options.allow_growth = True
tf.compat.v1.Session(config=tf_config)


def combine_conv_bn(model, conv_name, bn_name):
    conv_kxk_weights = model.get_layer(conv_name).get_weights()
    if len(conv_kxk_weights) > 1:
        conv_kxk_weights, conv_kxk_bias = conv_kxk_weights
    else:
        conv_kxk_weights = conv_kxk_weights[0]
        conv_kxk_bias = np.zeros((conv_kxk_weights.shape[-1],))

    gammas, betas, means, var = model.get_layer(bn_name).get_weights()
    var = var + 1e-3
    weight = (gammas * conv_kxk_weights / np.sqrt(var))
    bias = (((conv_kxk_bias - means) * gammas) / np.sqrt(var)) + betas
    model.get_layer(conv_name).set_weights([weight, bias])
    model = delete_layer(model, model.get_layer(bn_name), copy=True)
    return model


def combine_conv_compact(model, conv_name, compact_name):
    conv_kxk_weights = model.get_layer(conv_name).get_weights()
    if len(conv_kxk_weights) > 1:
        conv_kxk_weights, conv_kxk_bias = conv_kxk_weights
    else:
        conv_kxk_weights = conv_kxk_weights[0]
        conv_kxk_bias = np.zeros((conv_kxk_weights.shape[-1],))

    compact_kxk_weights = model.get_layer(compact_name).get_weights()
    compact_kxk_mask = compact_kxk_weights[1]
    compact_kxk_weights = compact_kxk_weights[0]
    conv_kxk_weights = conv_kxk_weights
    conv_kxk_bias = np.reshape(conv_kxk_bias, (1, 1, 1, conv_kxk_bias.shape[0]))
    with tf.Session() as sess:
        conv_1x1 = tf.convert_to_tensor(compact_kxk_weights)
        conv_kxk_w = tf.convert_to_tensor(conv_kxk_weights)
        conv_kxk_b = tf.convert_to_tensor(conv_kxk_bias)
        weight = K.conv2d(conv_kxk_w, conv_1x1, padding='same', data_format='channels_last').eval()
        bias = K.conv2d(conv_kxk_b, conv_1x1, padding='same', data_format='channels_last').eval()

    bias = np.sum(bias, axis=(0, 1, 2))
    model.get_layer(conv_name).set_weights([weight, bias])
    model = delete_layer(model, model.get_layer(compact_name), copy=True)
    del_channel = np.argwhere(compact_kxk_mask.reshape(-1) == 0)
    layer = model.get_layer(conv_name)
    model = delete_channels(model, layer, del_channel, copy=True)
    return model


def compactor_convert(model):
    layer_len = len(model.layers)
    layer_names = []
    for index, layer in enumerate(model.layers):
        layer_names.append(layer.name)
    for i in range(layer_len-2):
        current_name = layer_names[i]
        next_name = layer_names[i+1]
        last_name = layer_names[i+2]
        if ('conv' in current_name) and ('compactor' in next_name):
            model = combine_conv_compact(model, current_name, next_name)
        if ('conv' in current_name) and ('batch_normalization' in next_name):
            model = combine_conv_bn(model, current_name, next_name)
        if ('conv' in current_name) and ('batch_normalization' in next_name) and ('compactor' in last_name):
            model = combine_conv_compact(model, current_name, last_name)
    return model


def get_orignal_model():
    model = Sequential()
    model.add(Conv2D(10,
                     kernel_size=[3, 3],
                     input_shape=[28, 28, 1],
                     activation=None,
                     padding='same',
                     name='conv_1'))
    model.add(BatchNormalization(name='bn_1'))
    # model.add(CompactorLayer(10, kernel_regularizer=lasso(1e-4)))
    model.add(Activation('relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(10, [3, 3], activation=None, padding='same', name='conv_2'))
    model.add(BatchNormalization(name='bn_2'))
    # model.add(CompactorLayer(10, kernel_regularizer=lasso(1e-4)))
    model.add(Activation('relu'))
    model.add(Conv2D(10, [3, 3], activation=None, padding='same', name='conv_3'))
    model.add(BatchNormalization(name='bn_3'))
    # model.add(CompactorLayer(10, kernel_regularizer=lasso(1e-4)))
    model.add(Activation('relu'))
    model.add(Conv2D(10, [3, 3], activation=None, padding='same', name='conv_4'))
    model.add(BatchNormalization(name='bn_4'))
    # model.add(CompactorLayer(10, kernel_regularizer=lasso(1e-4)))
    model.add(Activation('relu'))
    model.add(MaxPool2D())
    model.add(layers.Permute((2, 1, 3)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', name='dense_1'))
    model.add(Dense(10, activation='softmax', name='dense_2'))
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def get_compact_modle():
    model = Sequential()
    model.add(Conv2D(10,
                     kernel_size=[3, 3],
                     input_shape=[28, 28, 1],
                     activation=None,
                     padding='same',
                     name='conv_1'))
    model.add(BatchNormalization(name='bn_1'))
    model.add(CompactorLayer(10, kernel_regularizer=lasso(1e-4)))
    model.add(Activation('relu'))
    model.add(MaxPool2D())
    model.add(Conv2D(10, [3, 3], activation=None, padding='same', name='conv_2'))
    model.add(BatchNormalization(name='bn_2'))
    model.add(CompactorLayer(10, kernel_regularizer=lasso(1e-4)))
    model.add(Activation('relu'))
    model.add(Conv2D(10, [3, 3], activation=None, padding='same', name='conv_3'))
    model.add(BatchNormalization(name='bn_3'))
    model.add(CompactorLayer(10, kernel_regularizer=lasso(1e-4)))
    model.add(Activation('relu'))
    model.add(Conv2D(10, [3, 3], activation=None, padding='same', name='conv_4'))
    model.add(BatchNormalization(name='bn_4'))
    model.add(CompactorLayer(10, kernel_regularizer=lasso(1e-4)))
    model.add(Activation('relu'))
    model.add(MaxPool2D())
    model.add(layers.Permute((2, 1, 3)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', name='dense_1'))
    model.add(Dense(10, activation='softmax', name='dense_2'))
    sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # Download data if needed and import.
    mnist = input_data.read_data_sets('tempData', one_hot=True, reshape=False)
    val_images = mnist.validation.images
    val_labels = mnist.validation.labels

    # train orignal model
    orig_model = get_orignal_model()
    results = orig_model.fit(mnist.train.images,
                        mnist.train.labels,
                        epochs=200,
                        batch_size=256,
                        verbose=2,
                        validation_data=(val_images, val_labels),
                        callbacks=[CompactorMonitor(step=2, patience=10, verbose=2)])

    orig_model.save('orignal_model.hdf5')

    # train compactor model
    compact_model = get_compact_modle()
    compact_model.load_weights('orignal_model.hdf5', by_name=True, skip_mismatch=True)

    results = compact_model.fit(mnist.train.images,
                        mnist.train.labels,
                        epochs=200,
                        batch_size=256,
                        verbose=2,
                        validation_data=(val_images, val_labels),
                        callbacks=[CompactorMonitor(min_flops=0.5, step=2, patience=10, verbose=2)])
    compact_model.save('compact_model.hdf5')
    loss = compact_model.evaluate(val_images, val_labels, batch_size=128, verbose=2)
    print('compact model loss:', loss, '\n')

    # test
    orig_model = load_model('orignal_model.hdf5')
    loss = orig_model.evaluate(val_images, val_labels, batch_size=128, verbose=2)
    print('orignal model loss:', loss, '\n')



if __name__ == '__main__':
    main()

