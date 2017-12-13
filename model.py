import numpy as np
import tensorflow as tf
import os
import logging
from utils import save_images, save_image
layers = tf.contrib.layers

def generate(z, n_img_pix, n_conv_hidden, n_channel, is_train=True, reuse=False):
    #logging.basicConfig(filename=os.path.join(base_dir,'conv_gen.log'), level=logging.INFO)

    n_repeat = int(np.log2(n_img_pix)) - 2
    #w_init = tf.random_normal_initializer(stddev=0.02)
    #gamma_init = tf.random_normal_initializer(1., 0.02) 

    l_conv =list()
    with tf.variable_scope("generator", reuse=reuse) as gen:
        n_output = int(np.prod([8, 8, n_conv_hidden]))
        x = layers.fully_connected(z, n_output, activation_fn=None)
        x = tf.reshape(x, [z.shape[0], 8, 8, n_conv_hidden])
        #x = tf.cast(x, tf.float32)
        #tf.nn.batchnormal
        for idx in range(n_repeat):
            x = layers.conv2d(x, n_conv_hidden, 3, 1, activation_fn=tf.nn.elu)
            l_conv.append(x)
            x = layers.conv2d(x, n_conv_hidden, 3, 1, activation_fn=tf.nn.elu)
            l_conv.append(x)
            if idx < n_repeat - 1:
                _,h,w,_ = x.shape
                x = tf.image.resize_nearest_neighbor(x, (h.value*2, w.value*2))
                l_conv.append(x)

        out = layers.conv2d(x, n_channel, 3, 1, activation_fn=None)
        l_conv.append(out)
        #logits = tf.nn.tanh(out)
    g_vars = tf.contrib.framework.get_variables(gen)
    return out, g_vars,l_conv


def encode(x, n_z, n_img_pix, n_conv_hidden, is_train=True, reuse=False):
    
    n_repeat = int(np.log2(n_img_pix)) - 2
    l_conv =list()
    #x = tf.cast(x, tf.float32)
    with tf.variable_scope("encoder", reuse=reuse) as enc:
        x = layers.conv2d(x, n_conv_hidden, 3, 1, activation_fn=tf.nn.elu)
        l_conv.append(x)
            
        for idx in range(n_repeat):
            n_channel = n_repeat * (idx + 1)
            x = layers.conv2d(x, n_channel, 3, 1, activation_fn=tf.nn.elu)
            l_conv.append(x)
            x = layers.conv2d(x, n_channel, 3, 1, activation_fn=tf.nn.elu)
            l_conv.append(x)
            if idx < n_repeat - 1:
                x = layers.conv2d(x, n_channel, 3, 2, activation_fn=tf.nn.elu)
                l_conv.append(x)

        x = tf.reshape(x, [-1, np.prod([8, 8, n_channel])])
        out = layers.fully_connected(x, n_z, activation_fn=None)
        #logits = tf.nn.sigmoid(out)
    e_vars = tf.contrib.framework.get_variables(enc)
    return out, e_vars, l_conv   


def decode(x, n_z, n_img_pix, n_conv_hidden, n_channel,  is_train=True, reuse=False):
    
    n_repeat = int(np.log2(n_img_pix)) - 2
    n_output = int(np.prod([8, 8, n_conv_hidden]))
    l_conv=list()
  
    #x = tf.cast(x, tf.float32)
    with tf.variable_scope("decoder", reuse=reuse) as dec:
        x = layers.fully_connected(x, n_output, activation_fn=None)
        x = tf.reshape(x, [x.shape[0].value, 8, 8, n_conv_hidden])
        
        for idx in range(n_repeat):
            x = layers.conv2d(x, n_conv_hidden, 3, 1, activation_fn=tf.nn.elu)
            l_conv.append(x)
            x = layers.conv2d(x, n_conv_hidden, 3, 1, activation_fn=tf.nn.elu)
            l_conv.append(x)
            if idx < n_repeat - 1:
                _,h,w,_ = x.shape
                x = tf.image.resize_nearest_neighbor(x, (h.value*2, w.value*2))
                l_conv.append(x)

        out = layers.conv2d(x, n_channel, 3, 1, activation_fn=None)
        l_conv.append(out)
        #logits = tf.nn.tanh(out)
        
    d_vars = tf.contrib.framework.get_variables(dec)
    return out, d_vars, l_conv
