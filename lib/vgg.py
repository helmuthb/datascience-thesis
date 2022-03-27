import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import MaxPool2D, Conv2D
from tensorflow.keras.applications.vgg16 import (
    VGG16, preprocess_input as vgg16_prep)


def vgg16_model(input_tensor: tf.Tensor) -> tuple:
    weight_layers = VGG16(include_top=True)
    prep = vgg16_prep
    # recreate vgg16 layers but using a slightly different config
    x = input_tensor
    blocks = [
        {'filters': 64, 'pool': True, 'layers': 2},
        {'filters': 128, 'pool': True, 'layers': 2},
        {'filters': 256, 'pool': True, 'layers': 3},
        {'filters': 512, 'pool': True, 'layers': 3},
        {'filters': 512, 'pool': False, 'layers': 3},
    ]
    for i, block in enumerate(blocks):
        block_name = f'block{i+1}'
        for j in range(block['layers']):
            name = f'{block_name}_conv{j+1}'
            conv = Conv2D(
                filters=block['filters'],
                kernel_size=(3, 3),
                name=name,
                padding='same',
                activation='relu'
            )
            x = conv(x)
            # copy weights over from trained model
            weights = weight_layers.get_layer(name).get_weights()
            conv.set_weights(weights)
        if block['pool']:
            pool = MaxPool2D(
                pool_size=2,
                strides=2,
                name=f'{block_name}_pool',
                padding='same'
            )
            x = pool(x)
    # adding some more layers on top
    maxpool = MaxPool2D(
        pool_size=(3, 3),
        strides=1,
        padding='same',
        name='block5_pool'
    )
    maxpool_out = maxpool(x)
    conv6_1 = Conv2D(
        filters=1024,
        kernel_size=3,
        dilation_rate=6,
        padding='same',
        activation='relu',
        name='block6_conv1'
    )
    conv6_1_out = conv6_1(maxpool_out)
    # set weights randomly chosen from first FC top layer
    (fc1_weights, fc1_biases) = weight_layers.get_layer('fc1').get_weights()
    conv6_1_weights = np.random.choice(
        np.reshape(fc1_weights, (-1,)),
        (3, 3, 512, 1024)
    )
    conv6_1_biases = np.random.choice(fc1_biases, (1024,))
    conv6_1.set_weights([conv6_1_weights, conv6_1_biases])
    conv6_2 = Conv2D(
        filters=1024,
        kernel_size=1,
        padding='same',
        activation='relu',
        name='block6_conv2'
    )
    conv6_2_out = conv6_2(conv6_1_out)
    # set weights randomly chosen from second FC top layer
    (fc2_weights, fc2_biases) = weight_layers.get_layer('fc2').get_weights()
    conv6_2_weights = np.random.choice(
        np.reshape(fc2_weights, (-1,)),
        (1, 1, 1024, 1024)
    )
    conv6_2_biases = np.random.choice(fc2_biases, (1024,))
    conv6_2.set_weights([conv6_2_weights, conv6_2_biases])
    return (tf.keras.Model(inputs=input_tensor, outputs=conv6_2_out), prep)
