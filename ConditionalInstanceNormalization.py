#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_probability as tfp;

class ConditionalInstanceNormalization(tf.keras.layers.Layer):
    
    def __init__(self, num_categories):
        
        super(ConditionalInstanceNormalization, self).__init__();
        assert type(num_categories) is int;
        self.num_categories = num_categories;

    def build(self, input_shape):
       
        # shape = (class_num, channels)
        shape = tf.TensorShape([self.num_categories]).concatenate(input_shape[-1:]);
        self.gamma = self.add_weight(name = 'gamma', shape = shape, initializer = tf.ones_initializer(), trainable = True);
        self.beta = self.add_weight(name = 'beta', shape = shape, initializer = tf.zeros_initializer(), trainable = True);
        super(ConditionalInstanceNormalization, self).build(input_shape);
        
    def call(self, inputs, labels):
        
        inputs_rank = inputs.get_shape().ndims;
        if inputs_rank != 4:
            raise ValueError('Input %s is not a 4D tensor.' % inputs.name);
        mean, variance = tf.nn.moments(inputs, axes = (1,2), keepdims = True);
        # gamme.shape = (batch, channels)
        gamma = tf.gather_nd(self.gamma, labels);
        # beta.shape = (batch, channels)
        beta = tf.gather_nd(self.beta, labels);
        # gamma.shape = (batch, 1, 1, channels) for broadcasting on h and w
        gamma = tf.expand_dims(tf.expand_dims(gamma,1),1);
        # beta.shape = (batch, 1, 1, channels) for broadcasting on h and w
        beta = tf.expand_dims(tf.expand_dims(beta,1),1);
        variance_epsilon = 1e-5;
        outputs = tf.nn.batch_normalization(inputs,mean,variance,beta,gamma,variance_epsilon);
        outputs.set_shape(inputs.get_shape());
        return outputs;
    
    def compute_output_shape(self, input_shape):
        
        return input_shape;
