#!/usr/bin/python3

import cv2;
import tensorflow as tf;
import tensorflow_probability as tfp;

class Encoder(tf.keras.Model):
    
    def __init__(self, encode_size = 16,  base_depth = 32):
        
        # encode_size: latent variable dimension
        super(Encoder,self).__init__();
        
        self.prior = tfp.distributions.Independent(tfp.distributions.Normal(loc = tf.zeros(encode_size), scale = 1), reinterpreted_batch_ndims = 1);
        
        self.normalize = tf.keras.layers.Lambda(lambda x: tf.cast(x,tf.float32) - 0.5);
        self.conv1 = tf.keras.layers.Conv2D(filters = base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same');
        self.relu1 = tf.keras.layers.LeakyReLU();
        self.conv2 = tf.keras.layers.Conv2D(filters = base_depth, kernel_size = (5,5), strides = (2,2), padding = 'same');
        self.relu2 = tf.keras.layers.LeakyReLU();
        self.conv3 = tf.keras.layers.Conv2D(filters = 2 * base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same');
        self.relu3 = tf.keras.layers.LeakyReLU();
        self.conv4 = tf.keras.layers.Conv2D(filters = 2 * base_depth, kernel_size = (5,5), strides = (2,2), padding = 'same');
        self.relu4 = tf.keras.layers.LeakyReLU();
        self.conv5 = tf.keras.layers.Conv2D(filters = 4 * base_depth, kernel_size = (7,7), strides = (1,1), padding = 'valid');
        self.relu5 = tf.keras.layers.LeakyReLU();
        self.flatten = tf.keras.layers.Flatten();
        self.dense = tf.keras.layers.Dense(units = tfp.layers.MultivariateNormalTriL.params_size(encode_size));
        self.gaussian = tfp.layers.MultivariateNormalTriL(encode_size, activity_regularizer = tfp.layers.KLDivergenceRegularizer(self.prior, weight=1.0));
        
    def call(self, inputs):
        
        result = self.normalize(inputs);
        result = self.conv1(result);
        result = self.relu1(result);
        result = self.conv2(result);
        result = self.relu2(result);
        result = self.conv3(result);
        result = self.relu3(result);
        result = self.conv4(result);
        result = self.relu4(result);
        result = self.conv5(result);
        result = self.relu5(result);
        result = self.flatten(result);
        result = self.dense(result);
        result = self.gaussian(result);
        assert isinstance(result, tfp.distributions.Distribution);
        return result;

class Decoder(tf.keras.Model):
    
    def __init__(self, input_shape = (28, 28, 1), encode_size = 16, base_depth = 32):
        
        super(Decoder,self).__init__();
        
        self.reshape = tf.keras.layers.Reshape([1, 1, encode_size]);
        self.conv1 = tf.keras.layers.Conv2DTranspose(filters = 2 * base_depth, kernel_size = (7,7), strides = (1,1), padding = 'valid');
        self.relu1 = tf.keras.layers.LeakyReLU();
        self.conv2 = tf.keras.layers.Conv2DTranspose(filters = 2 * base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same');
        self.relu2 = tf.keras.layers.LeakyReLU();
        self.conv3 = tf.keras.layers.Conv2DTranspose(filters = 2 * base_depth, kernel_size = (5,5), strides = (2,2), padding = 'same');
        self.relu3 = tf.keras.layers.LeakyReLU();
        self.conv4 = tf.keras.layers.Conv2DTranspose(filters = base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same');
        self.relu4 = tf.keras.layers.LeakyReLU();
        self.conv5 = tf.keras.layers.Conv2DTranspose(filters = base_depth, kernel_size = (5,5), strides = (2,2), padding = 'same');
        self.relu5 = tf.keras.layers.LeakyReLU();
        self.conv6 = tf.keras.layers.Conv2DTranspose(filters = base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same');
        self.relu6 = tf.keras.layers.LeakyReLU();
        self.conv7 = tf.keras.layers.Conv2D(filters = 1, kernel_size = (5,5), strides = (1,1), padding = 'same');
        self.flatten = tf.keras.layers.Flatten();
        self.bernoulli = tfp.layers.IndependentBernoulli(event_shape = input_shape, convert_to_tensor_fn = tfp.distributions.Bernoulli.logits);
        
    def call(self, inputs):
        
        result = self.reshape(inputs);
        result = self.conv1(result);
        result = self.relu1(result);
        result = self.conv2(result);
        result = self.relu2(result);
        result = self.conv3(result);
        result = self.relu3(result);
        result = self.conv4(result);
        result = self.relu4(result);
        result = self.conv5(result);
        result = self.relu5(result);
        result = self.conv6(result);
        result = self.relu6(result);
        result = self.conv7(result);
        result = self.flatten(result);
        result = self.bernoulli(result);
        assert isinstance(result, tfp.distributions.Distribution);
        return result;

class VAE(tf.keras.Model):
    
    def __init__(self, input_shape = (28, 28, 1), encode_size = 16, base_depth = 32):
        
        super(VAE, self).__init__();
        
        self.prior = tfp.distributions.Independent(tfp.distributions.Normal(loc = tf.zeros(encode_size), scale = 1));
        
        self.encoder = Encoder(encode_size, base_depth);
        self.decoder = Decoder(input_shape, encode_size, base_depth);
    
    def call(self, inputs, weight = 1.0):
        
        code_distr = self.encoder(inputs);
        code = code_distr.sample();
        sample_distr = self.decoder(code);
        likelihood_loss = -sample_distr.log_prob(inputs);

        return likelihood_loss;
    
    def sample(self, batch_size = 1):
        
        code = self.prior.sample(sample_shape = (batch_size));
        sample_distr = self.decoder(code);
        return sample_distr.sample();

if __name__ == "__main__":
    
    optimizer = tf.keras.optimizers.Adam(1e-4);
    vae = VAE();
    checkpoint = tf.train.Checkpoint(model = vae, optimizer = optimizer, optimizer_step = optimizer.iterations);
    checkpoint.restore(tf.train.latest_checkpoint('vae_checkpoints'));
    for i in range(10):
        img = vae.sample();
        img = img[0,...].numpy() * 255.0;
        cv2.imshow(str(i),img.astype('uint8'));
    cv2.waitKey();

