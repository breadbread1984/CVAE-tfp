#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_probability as tfp;
from ConditionalInstanceNormalization import ConditionalInstanceNormalization;

class Encoder(tf.keras.Model):
    
    def __init__(self, encode_size = 16, class_num = 10, base_depth = 32):
        
        # encode_size: latent variable dimension
        super(Encoder,self).__init__();
        
        self.class_num = class_num;
        self.prior = tfp.distributions.Independent(tfp.distributions.Normal(loc = tf.zeros(encode_size), scale = 1), reinterpreted_batch_ndims = 1);
        
        self.normalize = tf.keras.layers.Lambda(lambda x: tf.cast(x,tf.float32) - 0.5);
        self.conv1 = tf.keras.layers.Conv2D(filters = base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same');
        self.bn1 = ConditionalInstanceNormalization(class_num);
        self.relu1 = tf.keras.layers.LeakyReLU();
        self.conv2 = tf.keras.layers.Conv2D(filters = base_depth, kernel_size = (5,5), strides = (2,2), padding = 'same');
        self.bn2 = ConditionalInstanceNormalization(class_num);
        self.relu2 = tf.keras.layers.LeakyReLU();
        self.conv3 = tf.keras.layers.Conv2D(filters = 2 * base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same');
        self.bn3 = ConditionalInstanceNormalization(class_num);
        self.relu3 = tf.keras.layers.LeakyReLU();
        self.conv4 = tf.keras.layers.Conv2D(filters = 2 * base_depth, kernel_size = (5,5), strides = (2,2), padding = 'same');
        self.bn4 = ConditionalInstanceNormalization(class_num);
        self.relu4 = tf.keras.layers.LeakyReLU();
        self.conv5 = tf.keras.layers.Conv2D(filters = 4 * base_depth, kernel_size = (7,7), strides = (1,1), padding = 'valid');
        self.bn5 = ConditionalInstanceNormalization(class_num);
        self.relu5 = tf.keras.layers.LeakyReLU();
        self.flatten = tf.keras.layers.Flatten();
        self.dense = tf.keras.layers.Dense(units = tfp.layers.MultivariateNormalTriL.params_size(encode_size));
        self.gaussian = tfp.layers.MultivariateNormalTriL(encode_size, activity_regularizer = tfp.layers.KLDivergenceRegularizer(self.prior, weight=1.0));
        
    def call(self, inputs, labels):
        
        result = self.normalize(inputs);
        result = self.conv1(result);
        result = self.bn1(result,labels);
        result = self.relu1(result);
        result = self.conv2(result);
        result = self.bn2(result,labels);
        result = self.relu2(result);
        result = self.conv3(result);
        result = self.bn3(result,labels);
        result = self.relu3(result);
        result = self.conv4(result);
        result = self.bn4(result,labels);
        result = self.relu4(result);
        result = self.conv5(result);
        result = self.bn5(result,labels);
        result = self.relu5(result);
        result = self.flatten(result);
        result = self.dense(result);
        result = self.gaussian(result);
        assert isinstance(result, tfp.distributions.Distribution);
        return result;

class Decoder(tf.keras.Model):
    
    def __init__(self, input_shape = (28, 28, 1), encode_size = 16, class_num = 10, base_depth = 32):
        
        super(Decoder,self).__init__();
        
        self.class_num = class_num;
        
        self.reshape = tf.keras.layers.Reshape([1, 1, encode_size]);
        self.conv1 = tf.keras.layers.Conv2DTranspose(filters = 2 * base_depth, kernel_size = (7,7), strides = (1,1), padding = 'valid');
        self.bn1 = ConditionalInstanceNormalization(class_num);
        self.relu1 = tf.keras.layers.LeakyReLU();
        self.conv2 = tf.keras.layers.Conv2DTranspose(filters = 2 * base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same');
        self.bn2 = ConditionalInstanceNormalization(class_num);
        self.relu2 = tf.keras.layers.LeakyReLU();
        self.conv3 = tf.keras.layers.Conv2DTranspose(filters = 2 * base_depth, kernel_size = (5,5), strides = (2,2), padding = 'same');
        self.bn3 = ConditionalInstanceNormalization(class_num);
        self.relu3 = tf.keras.layers.LeakyReLU();
        self.conv4 = tf.keras.layers.Conv2DTranspose(filters = base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same');
        self.bn4 = ConditionalInstanceNormalization(class_num);
        self.relu4 = tf.keras.layers.LeakyReLU();
        self.conv5 = tf.keras.layers.Conv2DTranspose(filters = base_depth, kernel_size = (5,5), strides = (2,2), padding = 'same');
        self.bn5 = ConditionalInstanceNormalization(class_num);
        self.relu5 = tf.keras.layers.LeakyReLU();
        self.conv6 = tf.keras.layers.Conv2DTranspose(filters = base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same');
        self.bn6 = ConditionalInstanceNormalization(class_num);
        self.relu6 = tf.keras.layers.LeakyReLU();
        self.conv7 = tf.keras.layers.Conv2D(filters = 1, kernel_size = (5,5), strides = (1,1), padding = 'same');
        self.flatten = tf.keras.layers.Flatten();
        self.bernoulli = tfp.layers.IndependentBernoulli(event_shape = input_shape, convert_to_tensor_fn = tfp.distributions.Bernoulli.logits);
        
    def call(self, inputs, labels):
        
        result = self.reshape(inputs);
        result = self.conv1(result);
        result = self.bn1(result, labels);
        result = self.relu1(result);
        result = self.conv2(result);
        result = self.bn2(result, labels);
        result = self.relu2(result);
        result = self.conv3(result);
        result = self.bn3(result, labels);
        result = self.relu3(result);
        result = self.conv4(result);
        result = self.bn4(result, labels);
        result = self.relu4(result);
        result = self.conv5(result);
        result = self.bn5(result, labels);
        result = self.relu5(result);
        result = self.conv6(result);
        result = self.bn6(result, labels);
        result = self.relu6(result);
        result = self.conv7(result);
        result = self.flatten(result);
        result = self.bernoulli(result);
        assert isinstance(result, tfp.distributions.Distribution);
        return result;

class CVAE(tf.keras.Model):
    
    def __init__(self, input_shape = (28, 28, 1), encode_size = 16, class_num = 10, base_depth = 32):
        
        super(CVAE, self).__init__();
        
        self.class_num = class_num;
        self.prior = tfp.distributions.Independent(tfp.distributions.Normal(loc = tf.zeros(encode_size), scale = 1));
        
        self.encoder = Encoder(encode_size, class_num, base_depth);
        self.decoder = Decoder(input_shape, encode_size, class_num, base_depth);
    
    def call(self, inputs, labels, weight = 1.0):
        
        code_distr = self.encoder(inputs, labels);
        code = code_distr.sample();
        sample_distr = self.decoder(code, labels);
        likelihood_loss = -sample_distr.log_prob(inputs);

        return likelihood_loss;
    
    def sample(self, labels, batch_size = 1):
        
        code = self.prior.sample(sample_shape = (batch_size));
        sample_distr = self.decoder(code, labels);
        return sample_distr.sample();

if __name__ == "__main__":
    
    cvae = CVAE();
