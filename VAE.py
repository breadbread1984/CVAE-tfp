#!/usr/bin/python3

import cv2;
import tensorflow as tf;
import tensorflow_probability as tfp;

def Encoder(input_shape, encode_size = 16, base_depth = 32):

    inputs = tf.keras.Input(input_shape);
    results = tf.keras.layers.Lambda(lambda x: tf.cast(x,tf.float32) - 0.5)(inputs);
    results = tf.keras.layers.Conv2D(filters = base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same')(results);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2D(filters = base_depth, kernel_size = (5,5), strides = (2,2), padding = 'same')(results);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2D(filters = 2 * base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same')(results);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2D(filters = 2 * base_depth, kernel_size = (5,5), strides = (2,2), padding = 'same')(results);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2D(filters = 4 * base_depth, kernel_size = (7,7), strides = (1,1), padding = 'valid')(results);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Flatten()(results);
    results = tf.keras.layers.Dense(units = tfp.layers.MultivariateNormalTriL.params_size(encode_size))(results);
    prior = tfp.distributions.Independent(tfp.distributions.Normal(loc = tf.zeros(encode_size), scale = 1), reinterpreted_batch_ndims = 1);
    results = tfp.layers.MultivariateNormalTriL(encode_size, activity_regularizer = tfp.layers.KLDivergenceRegularizer(prior, weight = 1.0))(results);
    return tf.keras.Model(inputs = inputs, outputs = results);

def Decoder(output_shape = (28, 28, 1), encode_size = 16, base_depth = 32):

    inputs = tf.keras.Input((encode_size,));
    results = tf.keras.layers.Reshape((1, 1, encode_size))(inputs);
    results = tf.keras.layers.Conv2DTranspose(filters = 2 * base_depth, kernel_size = (7,7), strides = (1,1), padding = 'valid')(results);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2DTranspose(filters = 2 * base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same')(results);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2DTranspose(filters = 2 * base_depth, kernel_size = (5,5), strides = (2,2), padding = 'same')(results);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2DTranspose(filters = base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same')(results);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2DTranspose(filters = base_depth, kernel_size = (5,5), strides = (2,2), padding = 'same')(results);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2DTranspose(filters = base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same')(results);
    results = tf.keras.layers.BatchNormalization()(results);
    results = tf.keras.layers.LeakyReLU()(results);
    results = tf.keras.layers.Conv2D(filters = 1, kernel_size = (5,5), strides = (1,1), padding = 'same')(results);
    results = tf.keras.layers.Flatten()(results);
    results = tfp.layers.IndependentBernoulli(event_shape = output_shape, convert_to_tensor_fn = tfp.distributions.Bernoulli.logits)(results);
    return tf.keras.Model(inputs = inputs, outputs = results);

class VAE(tf.keras.Model):
    
    def __init__(self, input_shape = (28, 28, 1), encode_size = 16, base_depth = 32):
        
        super(VAE, self).__init__();
        
        self.prior = tfp.distributions.Independent(tfp.distributions.Normal(loc = tf.zeros(encode_size), scale = 1));
        
        self.encoder = Encoder(input_shape, encode_size, base_depth);
        self.decoder = Decoder(input_shape, encode_size, base_depth);
    
    def call(self, inputs):
        
        code = self.encoder(inputs);
        sample = self.decoder(code);
        return sample;
    
    def sample(self, batch_size = 1):
        
        code = self.prior.sample(sample_shape = (batch_size,));
        sample = self.decoder(code);
        retval = tf.clip_by_value(sample.sample(),0,1) * 255.;
        retval = tf.cast(retval, dtype = tf.uint8);
        return retval;

if __name__ == "__main__":
    
    vae = VAE();
    vae.load_weights('vae.h5');
    for i in range(10):
        img = vae.sample()[0,...];
        cv2.imshow(str(i),img.numpy());
    cv2.waitKey();

