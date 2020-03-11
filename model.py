from __future__ import absolute_import, division, print_function
import os
import glob
import cv2
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.layers import Conv2D, Input, Layer, Dense, Flatten, Conv2DTranspose, ReLU, LeakyReLU, Dropout, AveragePooling2D, LayerNormalization, Activation, Softmax
from tensorflow.keras.activations import tanh
from tensorflow.keras.models import Model,Sequential
import tensorflow_addons as tfa
import numpy as np
import random
import sys
import math



class MultiHeadAttention(Model):
    def __init__(self, batch_size: int, head_num: int, dim: int, length: int, dropout_rate: float = 0.5):
        self.dense_query = Dense(dim)
        self.dense_key = Dense(dim)
        self.dense_value = Dense(dim)
        self.softmax = Softmax()
        self.dropot = Dropout(dropout_rate)
        

    def call(self, inputs, attention_mask):
        query = self.dense_query(inputs)
        key = self.dense_key(inputs)
        value = self.dense_value(inputs)
        query_split, key_split, value_split = self._split(query), self._split(key), self._split(value)
        key_split = tf.transpose(key_split, perm=[0, 1, 3, 2])
        query_value = tf.matmul(query_split, key_split)
        query_value *= ((self.dim // self.head_num)**(-0.5))
        
        mask_expand = tf.expand_dims(attention_mask, axis=1)
        #マスクされたところが1のため、大きいマイナスの値を付与することで、
        #softmaxにおいて影響が無いようにする
        mask_expand *= -10 * 6
        query_value = self.softmax(query_value + mask_expand)
        

        
        
            
        
    def _split(self, inputs):
        with tf.name_scope('split_attention'):
            #分割(分割後の次元は(batch_size*length,head_num,dim/head_num))
            split_inputs = tf.stack([tf.split(inputs, num_or_size_splits=self.head_num, axis=-1)])[0]
            #分割(分割後の次元は(batch_size,length,head_num,dim/head_num))
            split_inputs = tf.stack([tf.split(inputs, num_or_size_splits=self.batch_size, axis=-1)])[0]
            #次元を(batch_size,head_num,length,dim/head_num)に変更
            split_inputs = tf.transpose(split_inputs, perm=[0, 2, 1, 3])
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