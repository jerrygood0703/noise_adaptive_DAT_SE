from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow import keras as keras
import numpy as np
from hyperparameters import Hyperparams as hp

class LSTM_SE(object):
    def __init__(self, frame_size, name="generator"):
        self.name = name
        self.frame_size = frame_size
    def __call__(self, x, reuse=True, is_training=True):

        with tf.variable_scope(self.name, reuse=reuse) as vs:
            inputs = x # [batch, 1, freq, frame_size]             
            with tf.variable_scope("Encoder"):    
                h_e = tf.squeeze(tf.transpose(inputs, [0,3,2,1]), -1)
                h_e = keras.layers.Bidirectional(keras.layers.LSTM(units=512, return_sequences=True, unroll=True))(h_e)

            with tf.variable_scope("Decoder"):    
                h_d = keras.layers.Bidirectional(keras.layers.LSTM(units=512, return_sequences=True, unroll=True))(h_e)
                h_d = keras.layers.Dense(units=hp.f_bin, activation=None)(h_d)
                h_d = tf.expand_dims(tf.transpose(h_d, [0,2,1]), axis=1)
        return h_d, h_e

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name == var.name.split('/')[0]]

    @property
    def vars_enc(self):
        return [var for var in tf.global_variables() if (self.name == var.name.split('/')[0] and var.name.split('/')[1] == 'Encoder')]

    @property
    def vars_dec(self):
        return [var for var in tf.global_variables() if (self.name == var.name.split('/')[0] and var.name.split('/')[1] == 'Decoder')]

class LSTM_Cls(object):
    def __init__(self, frame_size, NOISETYPES, name="classifier"):
        self.name = name
        self.NOISETYPES = NOISETYPES
        self.frame_size = frame_size
    def __call__(self, h_e, reuse=True, is_training=True):

        with tf.variable_scope(self.name, reuse=reuse) as vs:
            # h_e = [batch, T, C] 
            h_c = keras.layers.LSTM(units=1024, return_sequences=False, unroll=True)(h_e)
            phn_logits = keras.layers.Dense(units=self.NOISETYPES, activation=None)(h_c)

        return phn_logits

    @property
    def vars(self):
        return [var for var in tf.global_variables() if self.name == var.name.split('/')[0]]
