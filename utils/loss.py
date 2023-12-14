import tensorflow as tf
from keras import backend as K
import numpy as np

def loss_mae_ssim(y_true, y_pred):
    pic_para = 1 # 1e-2
    mae_para = 0.84
    ssim_para=0.1
    # nomolization
    x = y_true
    y = y_pred
    x = (x - K.min(x)) / (K.max(x) - K.min(x))
    y = (y - K.min(y)) / (K.max(y) - K.min(y))

    ssim_loss = ssim_para * (1 - K.mean(tf.image.ssim(x, y, 1)))
    mae_loss = mae_para * K.mean(K.square(y - x))
    pic_loss=pic_para*K.mean(abs(y-x))
    return pic_loss+ssim_loss

def loss_mse_ssim(y_true, y_pred):
    ssim_para = 0.1 #1e-1 # 1e-2
    mse_para = 1

    # nomolization
    x = y_true
    y = y_pred
    x = (x - K.min(x)) / (K.max(x) - K.min(x))
    y = (y - K.min(y)) / (K.max(y) - K.min(y))
    # output = 0.1 * y / K.max(K.sum(K.abs(y)))
    ssim_loss = ssim_para * (1 - K.mean(tf.image.ssim(x, y, 1)))
    mse_loss = mse_para * K.mean(K.square(y - x))

    return mse_loss + ssim_loss

def loss_mae_mse(y_true, y_pred):
    mae_para = 0.2
    mse_para = 1

    # nomolization
    x = y_true
    y = y_pred
    x = (x - K.min(x)) / (K.max(x) - K.min(x))
    y = (y - K.min(y)) / (K.max(y) - K.min(y))

    mae_loss = mae_para * K.mean(K.abs(x-y))
    mse_loss = mse_para * K.mean(K.square(y - x))

    return mae_loss + mse_loss

def loss_mse_mae_ssim(y_true, y_pred):
    mae_para = 0.2
    mse_para = 1
    ssim_para = 1e-1
    # nomolization
    x = y_true
    y = y_pred
    x = (x - K.min(x)) / (K.max(x) - K.min(x))
    y = (y - K.min(y)) / (K.max(y) - K.min(y))

    mae_loss = mae_para * K.mean(K.abs(x-y))
    mse_loss = mse_para * K.mean(K.square(y - x))
    ssim_loss = ssim_para * (1 - K.mean(tf.image.ssim(x, y, 1)))

    return mae_loss + mse_loss + ssim_loss
def loss_mse(y_true, y_pred):
    mae_para = 0
    mse_para = 1
    ssim_para = 0
    # nomolization
    x = y_true
    y = y_pred
    x = (x - K.min(x)) / (K.max(x) - K.min(x))
    y = (y - K.min(y)) / (K.max(y) - K.min(y))

    mae_loss = mae_para * K.mean(K.abs(x-y))
    mse_loss = mse_para * K.mean(K.square(y - x))
    ssim_loss = ssim_para * (1 - K.mean(tf.image.ssim(x, y, 1)))

    return mse_loss