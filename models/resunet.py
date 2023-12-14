import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, Activation, UpSampling2D, Lambda, Dropout, MaxPooling2D, multiply, add,Conv2DTranspose
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from .common import fft2d, fftshift2d, gelu, pixel_shiffle, global_average_pooling2d
import numpy as np

def npifft2d(input,gamma=0.1):
    temp = K.permute_dimensions(input, (0, 3, 1, 2))

    a=tf.ifft2d(tf.complex(temp, tf.zeros_like(temp)))
    absfft = tf.pow(tf.abs(a) + 1e-8, gamma)

    output = K.permute_dimensions(absfft, (0, 2, 3, 1))
    return output

def FCALayer(input, channel, reduction=16):
    size_psc = input.get_shape().as_list()[1]
    absfft1 = Lambda(fft2d, arguments={'gamma': 0.8})(input)
    absfft1 = Lambda(fftshift2d, arguments={'size_psc': size_psc})(absfft1)
    absfft2 = Conv2D(channel, kernel_size=3, activation='relu', padding='same')(absfft1)
    W = Lambda(global_average_pooling2d)(absfft2)
    W = Conv2D(channel // reduction, kernel_size=1, activation='relu', padding='same')(W)
    W = Conv2D(channel, kernel_size=1, activation='sigmoid', padding='same')(W)
    mul = multiply([input, W])
    output = add([mul, input])
    return output

def AttnBlock2D(x, g, inter_channel, data_format='channels_last'):

    theta_x = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(x)
    phi_g = Conv2D(inter_channel, [1, 1], strides=[1, 1], data_format=data_format)(g)

    f = Activation('relu')(add([theta_x, phi_g]))

    psi_f = Conv2D(1, [1, 1], strides=[1, 1], data_format=data_format)(f)

    rate = Activation('sigmoid')(psi_f)

    att_x = multiply([x, rate])

    return att_x


def attention_up_and_concate(down_layer, layer, scale=2,data_format='channels_last'):

    if data_format == 'channels_last':
        in_channel = down_layer.get_shape().as_list()[3]
    else:
        in_channel = down_layer.get_shape().as_list()[1]

    up=Conv2DTranspose(in_channel,(4,4),strides=(2,2),padding='same',use_bias=False)(down_layer)

    if data_format == 'channels_last':
        # print(down_layer.shape())
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    else:
        my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=1))  # 参考代码这个地方写错了，x[1] 写成了 x[3]
    concate = my_concat([up, layer])
    return concate

# Attention U-Net
def att_unet(input_shape, size_psc=128,scale=2, data_format='channels_last'):
    # inputs = (3, 160, 160)
    inputs = Input(input_shape)
    depth = 4  #4
    features = 64
    skips = []

    conv = Conv2D(64, kernel_size=3,activation='relu', padding='same')(inputs)
    x= Lambda(gelu)(conv)
    # depth = 0, 1, 2, 3



    for i in range(depth):
        # ENCODER
        x1=Conv2D(features, kernel_size=1, padding='same', use_bias=False, data_format=data_format)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = add([x,x1])
        skips.append(x)
        x = MaxPooling2D((2, 2), data_format='channels_last')(x)
        features = features * 2

    # BOTTLENECK
    x1 = Conv2D(features, kernel_size=1, padding='same', use_bias=False, data_format=data_format)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x = Dropout(0.2)(x)
    x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
    x = add([x, x1])

    # DECODER
    for i in reversed(range(depth)): # reversed(range(4)) = [3, 2, 1, 0]; range(4) = [0, 1, 2, 3]
        features = features // 2
        x = attention_up_and_concate(x, skips[i], data_format=data_format)
        x1 = Conv2D(features, kernel_size=1, padding='same', use_bias=False, data_format=data_format)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = Dropout(0.2)(x)
        x = Conv2D(features, (3, 3), activation='relu', padding='same', data_format=data_format)(x)
        x = add([x, x1])
    n_label=1

    x=Conv2DTranspose(64,(4,4),strides=(2,2),padding='same',use_bias=False)(x)

    conv6 = Conv2D(n_label, (1, 1), padding='same', data_format=data_format)(x)
    conv7 = Activation('sigmoid')(conv6)

    model = Model(inputs=inputs, outputs=conv7)
    return model

