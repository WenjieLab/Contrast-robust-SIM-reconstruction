import argparse
import glob
import os
from time import *
import cv2

import imageio
import numpy as np
from PIL import Image
from keras import optimizers
from models import res_unet_plus

from src.models import resunet_up

os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
import tensorflow.compat.v1 as tf
from utils.utils import prctile_norm, rm_outliers
from skimage.restoration import rolling_ball

from utils.process_data import auto_brightness_contrast


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="./testdata")
parser.add_argument("--folder_test", type=str, default="raw")
parser.add_argument("--gpu_id", type=str, default="0")
parser.add_argument("--gpu_memory_fraction", type=float, default=0.5)
parser.add_argument("--model_name", type=str, default="model")
parser.add_argument("--model_weights", type=str, default='../weights/weights.best.h5')
parser.add_argument("--input_height", type=int, default=512) # 1024 2048
parser.add_argument("--input_width", type=int, default=512)
parser.add_argument("--scale_factor", type=int, default=2)


args = parser.parse_args()
gpu_id = args.gpu_id
gpu_memory_fraction = args.gpu_memory_fraction
data_dir = args.data_dir
folder_test = args.folder_test
model_name = args.model_name
model_weights = args.model_weights
input_width = args.input_width
input_height = args.input_height
scale_factor = args.scale_factor

output_name = 'output_' + folder_test + '-'+ model_name + '-'
test_images_path = data_dir + '/' + folder_test
output_dir = data_dir + '/' + output_name

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] ="1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
   try:
#     # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
       tf.config.experimental.set_memory_growth(gpu, True)
       logical_gpus = tf.config.experimental.list_logical_devices('GPU')
       print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
   except RuntimeError as e:
#     # Memory growth must be set before GPUs have been initialized
       print(e)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
tf.Session(config=tf.ConfigProto(allow_soft_placement=True))


# --------------------------------------------------------------------------------
#                              test data path
# --------------------------------------------------------------------------------
img_path = glob.glob(test_images_path + '/*.tif')
# img_path = glob.glob(test_images_path + '/*')
img_path.sort()

if img_path:
    flag_recon = 1
    img_path = glob.glob(test_images_path + '/*.tif')
    # img_path = glob.glob(test_images_path + '/*')
    img_path.sort()

    n_channel = 9
    # n_channel = len(glob.glob(img_path[0] + '/*.tif'))

    output_dir = output_dir + 'SIM'
    print(output_dir)

# --------------------------------------------------------------------------------
#                          select models and load weights
# --------------------------------------------------------------------------------

modelFns = {'model': resunet_up.att_unet}
modelFN = modelFns[model_name]

optimizer = optimizers.adam(lr=1e-4, beta_1=0.9, beta_2=0.999)

m = modelFN((input_height, input_width, n_channel), scale=scale_factor)

m.load_weights(model_weights)

m.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])

im_count = 0

#————————————————————————————————————————————————————————————————#
# 文件夹路径
img_path = test_images_path

# 获取文件夹中的所有图像文件
image_files = sorted([f for f in os.listdir(img_path) if f.endswith('.tif')])
for i in range(0, len(image_files), 9):
    # 读取九张图像
    img_batch = []
    for j in range(9):
        if i + j < len(image_files):
            image_path = os.path.join(img_path, image_files[i + j])
            img = np.array(imageio.imread(image_path).astype(np.float64))
            # image = Image.open(image_path)
            img = img[0:input_height, 0:input_height]
            img_batch.append(img)
    start_time = time()
    img = np.array(img_batch).transpose((1, 2, 0))
    print(img.shape)
    img = img[np.newaxis, :, :, :]

    img = prctile_norm(img)
    pr = rm_outliers(prctile_norm(np.squeeze(m.predict(img))))
    pr = np.uint16(pr * 65535)
    # 去背景
    background = rolling_ball(pr, radius=10)
    pr = pr - background

    img = Image.fromarray(pr)
    im_count = im_count + 1
    outName = output_dir + str(im_count) + '_outputSIM.tif'
    print(outName)
    img.save(outName)

end_time = time()
# print(end_time - start_time)
print("success")

