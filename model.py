from __future__ import absolute_import, division, print_function
import os
import glob
import cv2
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.layers import Conv2D, Input, Layer, Dense, Flatten, Conv2DTranspose, ReLU, LeakyReLU, Dropout, AveragePooling2D, LayerNormalization,Activation
from tensorflow.keras.activations import tanh
from tensorflow.keras.models import Model,Sequential
import tensorflow_addons as tfa
import numpy as np
import random
import sys
import math

class Attention(Model):
    def __init__(self, dim: int, batch_size: int, length: int, head_num: int):
        self.dim = dim
        self.batch_size = batch_size
        self.length = length
        self.head_num = head_num
        

    def call(self, inputs, attntion_mask):
        pass

class MultiHeadAttention(Model):
    def __init__(self, batch_size: int, head_num: int, dim: int, length: int):
        self.attentions = [Attention(dim / head_num, batch_size, length, head_num) for _ in range(head_num)]
        self.dense_query_key = Dense(dim)
        self.dense_value = Dense(dim)

    def call(self, inputs, attention_mask):
        query_key = self.dense_query_key(inputs)
        value = self.dense_value(inputs)
        
        for u in range(self.head_num):
            #分割したinputとattention maskを使い、計算
            continue
            
        
    def _split(self, inputs):
        with tf.name_scope('split_attention'):
            #分割(分割後の次元は(batch_size*length,head_num,dim/head_num))
            split_inputs = tf.stack([tf.split(inputs, num_or_size_splits=self.head_num, axis=-1)])[0]
            #次元を(head_num,batch_size*length,dim/head_num)に変更
            split_inputs = tf.transpose(split_inputs, perm=[2, 0, 1])
            return split_inputs

class FeedForward(Model):
    def __init__(self):
        pass


class Positional_Encoder(Model):
    def __init__(self, batch_size: int, max_length: int, dim: int, *args, **kwargs):
        #posを作成
        pos = np.arange(max_length)
        pos_metrix_twodim = np.tile(pos, (dim, 1))

        #10000**(2i/dmodel)を作成
        depth = np.arange(dim)
        depth_numerals = np.tile(depth.reshape(dim, 1), (1, max_length))
        depth_metrix = np.power(10000.0, (depth_numerals // 2 * 2) / dim)
        
        #奇数、偶数列を作成
        depth_for_oddeven = pos_metrix_twodim.ravel().reshape([max_length, dim], order="F")
        odd_numerals = depth_for_oddeven % 2
        even_numerals = np.array(depth_for_oddeven % 2 == 0, dtype=np.int32)
        
        #角度(2pi~10000*2pi)に変換
        degree_metrix = (pos_metrix_twodim / depth_metrix) * 2 * math.pi
        
        #変換
        self.potisional_sin = np.sin(np.dot(even_numerals,degree_metrix ))
        self.positional_cos = np.cos(np.dot(odd_numerals, degree_metrix))
        
    def call(self, inputs):
        return self.potisional_sin + self.positional_cos


class Transformer(Model):
    def __init__(self):
        pass

class BERT:
    def __init__(self):
        pass