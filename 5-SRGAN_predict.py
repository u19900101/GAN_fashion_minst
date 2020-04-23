from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, Add
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.applications import VGG19
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from data_loader import DataLoader
import numpy as np
import os

import keras.backend as K


class SRGAN():
    def __init__(self):
        # 低分辨率图的shape
        self.channels = 3
        self.lr_height = 128
        self.lr_width = 128
        self.lr_shape = (self.lr_height, self.lr_width, self.channels)
        # 高分辨率图的shape
        self.hr_height = self.lr_height * 4
        self.hr_width = self.lr_width * 4
        self.hr_shape = (self.hr_height, self.hr_width, self.channels)

        # 16个残差卷积块
        self.n_residual_blocks = 16
        # 优化器
        optimizer = Adam(0.0002, 0.5)

        # 数据集
        self.dataset_name = 'DIV'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.hr_height, self.hr_width))

        patch = int(self.hr_height / 2 ** 6)
        self.disc_patch = (patch, patch, 1)

        # 建立生成模型
        self.generator = self.build_generator()

    def build_generator(self):

        def residual_block(layer_input, filters):
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(layer_input)
            d = Activation('relu')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Conv2D(filters, kernel_size=3, strides=1, padding='same')(d)
            d = BatchNormalization(momentum=0.8)(d)
            d = Add()([d, layer_input])
            return d

        def deconv2d(layer_input):
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(256, kernel_size=3, strides=1, padding='same')(u)
            u = Activation('relu')(u)
            return u

        img_lr = Input(shape=self.lr_shape)
        # 第一部分，低分辨率图像进入后会经过一个卷积+RELU函数
        c1 = Conv2D(64, kernel_size=9, strides=1, padding='same')(img_lr)
        c1 = Activation('relu')(c1)

        # 第二部分，经过16个残差网络结构，每个残差网络内部包含两个卷积+标准化+RELU，还有一个残差边。
        r = residual_block(c1, 64)
        for _ in range(self.n_residual_blocks - 1):
            r = residual_block(r, 64)

        # 第三部分，上采样部分，将长宽进行放大，两次上采样后，变为原来的4倍，实现提高分辨率。
        c2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(r)
        c2 = BatchNormalization(momentum=0.8)(c2)
        c2 = Add()([c2, c1])
        u1 = deconv2d(c2)
        u2 = deconv2d(u1)
        gen_hr = Conv2D(self.channels, kernel_size=9, strides=1, padding='same', activation='tanh')(u2)

        return Model(img_lr, gen_hr)


if __name__ == '__main__':
    gan = SRGAN()
    gan.generator.load_weights("weights/DIV/gen_epoch500.h5",
                                skip_mismatch=True)
    imgs_hr, imgs_lr = gan.data_loader.load_data(1)

    fake_hr = gan.generator.predict(imgs_lr)
    # gan.train(epochs=6, init_epoch=0, batch_size=1, sample_interval=5)

