#!/usr/bin/python3

import cv2;
import tensorflow as tf;
import tensorflow_probability as tfp;
from ConditionalInstanceNormalization import ConditionalInstanceNormalization;

def Encoder(input_shape, cls_num, encode_size = 16, base_depth = 32):

    inputs = tf.keras.Input(input_shape);
    cls = tf.keras.Input((1), dtype = tf.int64);
    results = tf.keras.layers.Lambda(lambda x: tf.cast(x,tf.float32) - 0.5)(inputs);
    results = tf.keras.layers.Conv2D(filters = base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same')(results);
    results = ConditionalInstanceNormalization(cls_num)(results, cls);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2D(filters = base_depth, kernel_size = (5,5), strides = (2,2), padding = 'same')(results);
    results = ConditionalInstanceNormalization(cls_num)(results, cls);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2D(filters = 2 * base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same')(results);
    results = ConditionalInstanceNormalization(cls_num)(results, cls);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2D(filters = 2 * base_depth, kernel_size = (5,5), strides = (2,2), padding = 'same')(results);
    results = ConditionalInstanceNormalization(cls_num)(results, cls);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2D(filters = 4 * base_depth, kernel_size = (7,7), strides = (1,1), padding = 'valid')(results);
    results = ConditionalInstanceNormalization(cls_num)(results, cls);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Flatten()(results);
    results = tf.keras.layers.Dense(units = tfp.layers.MultivariateNormalTriL.params_size(encode_size))(results);
    prior = tfp.distributions.Independent(tfp.distributions.Normal(loc = tf.zeros(encode_size), scale = 1), reinterpreted_batch_ndims = 1);
    results = tfp.layers.MultivariateNormalTriL(encode_size, activity_regularizer = tfp.layers.KLDivergenceRegularizer(prior, weight = 1.0))(results);
    return tf.keras.Model(inputs = (inputs, cls), outputs = results);

def Decoder(output_shape, cls_num, encode_size = 16, base_depth = 32):

    inputs = tf.keras.Input((encode_size,));
    cls = tf.keras.Input((1), dtype = tf.int64);
    results = tf.keras.layers.Reshape((1, 1, encode_size))(inputs);
    results = tf.keras.layers.Conv2DTranspose(filters = 2 * base_depth, kernel_size = (7,7), strides = (1,1), padding = 'valid')(results);
    results = ConditionalInstanceNormalization(cls_num)(results, cls);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2DTranspose(filters = 2 * base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same')(results);
    results = ConditionalInstanceNormalization(cls_num)(results, cls);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2DTranspose(filters = 2 * base_depth, kernel_size = (5,5), strides = (2,2), padding = 'same')(results);
    results = ConditionalInstanceNormalization(cls_num)(results, cls);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2DTranspose(filters = base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same')(results);
    results = ConditionalInstanceNormalization(cls_num)(results, cls);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2DTranspose(filters = base_depth, kernel_size = (5,5), strides = (2,2), padding = 'same')(results);
    results = ConditionalInstanceNormalization(cls_num)(results, cls);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2DTranspose(filters = base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same')(results);
    results = ConditionalInstanceNormalization(cls_num)(results, cls);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2D(filters = 1, kernel_size = (5,5), strides = (1,1), padding = 'same')(results);
    results = tf.keras.layers.Flatten()(results);
    results = tfp.layers.IndependentBernoulli(event_shape = output_shape, convert_to_tensor_fn = tfp.distributions.Bernoulli.logits)(results);
    return tf.keras.Model(inputs = (inputs, cls), outputs = results);

class CVAE(tf.keras.Model):
    
    def __init__(self, input_shape = (28, 28, 1), cls_num = 10, encode_size = 16, base_depth = 32):
        
        super(CVAE, self).__init__();
        
        self.prior = tfp.distributions.Independent(tfp.distributions.Normal(loc = tf.zeros(encode_size), scale = 1));
        
        self.encoder = Encoder(input_shape, cls_num, encode_size, base_depth);
        self.decoder = Decoder(input_shape, cls_num, encode_size, base_depth);
    
    def call(self, inputs):
        
        img = inputs[0];
        cls = inputs[1];
        code = self.encoder((img, cls));
        sample = self.decoder((code, cls));
        return sample;
    
    def sample(self, cls, batch_size = 1):
        
        code = self.prior.sample(sample_shape = (batch_size,));
        sample = self.decoder((code, cls));
        retval = tf.clip_by_value(sample.sample(),0,1) * 255.;
        retval = tf.cast(retval, dtype = tf.uint8);
        return retval;

if __name__ == "__main__":
    
    cvae = CVAE();
    cvae.load_weights('cvae.h5');
    for i in range(10):
        img = cvae.sample(i)[0,...];
        cv2.imshow(str(i),img.numpy());
    cv2.waitKey();

