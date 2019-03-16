#!/usr/bin/python3

import tensorflow as tf;
import tensorflow_probability as tfp;

class Encoder(tf.keras.Model):
    
    def __init__(self, encode_size = 16, class_num = 10, base_depth = 32):
        
        super(Encoder,self).__init__();
        
        self.encode_size = encode_size;
        self.class_num = class_num;
        self.base_depth = base_depth;
        
        self.normalize = tf.keras.layers.Lambda(lambda x: tf.cast(x,tf.float32) - 0.5);
        self.conv1 = tf.keras.layers.Conv2D(filters = base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same');
        self.bn1 = [tf.keras.layers.BatchNormalization() for i in range(class_num)];
        self.relu1 = tf.keras.layers.LeakyReLU();
        self.conv2 = tf.keras.layers.Conv2D(filters = base_depth, kernel_size = (5,5), strides = (2,2), padding = 'same');
        self.bn2 = [tf.keras.layers.BatchNormalization() for i in range(class_num)];
        self.relu2 = tf.keras.layers.LeakyReLU();
        self.conv3 = tf.keras.layers.Conv2D(filters = 2 * base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same');
        self.bn3 = [tf.keras.layers.BatchNormalization() for i in range(class_num)];
        self.relu3 = tf.keras.layers.LeakyReLU();
        self.conv4 = tf.keras.layers.Conv2D(filters = 2 * base_depth, kernel_size = (5,5), strides = (2,2), padding = 'same');
        self.bn4 = [tf.keras.layers.BatchNormalization() for i in range(class_num)];
        self.relu4 = tf.keras.layers.LeakyReLU();
        self.conv5 = tf.keras.layers.Conv2D(filters = 4 * base_depth, kernel_size = (7,7), strides = (1,1), padding = 'valid');
        self.bn5 = [tf.keras.layers.BatchNormalization() for i in range(class_num)];
        self.relu5 = tf.keras.layers.LeakyReLU();
        self.flatten = tf.keras.layers.Flatten();
        self.dense = tf.keras.layers.Dense(tfp.layers.MultivariateNormalTril.params_size(encode_size));
        self.gaussian = tfp.layers.MultivariateNormalTril(encode_size);
        
    def call(self, input, label):
        
        result = self.normalize(input);
        result = self.conv1(result);
        result = self.bn1[label](result);
        result = self.relu1(result);
        result = self.conv2(result);
        result = self.bn2[label](result);
        result = self.relu2(result);
        result = self.conv3(result);
        result = self.bn3[label](result);
        result = self.relu3(result);
        result = self.conv4(result);
        result = self.bn4[label](result);
        result = self.relu4(result);
        result = self.conv5(result);
        result = self.bn5[label](result);
        result = self.relu5(result);
        result = self.flatten(result);
        result = self.dense(result);
        result = self.gaussian(result);
        assert isinstance(result, tfp.distributions.Distribution);
        return result;

class Decoder(tf.keras.Model):
    
    def __init__(self, input_shape = (28, 28, 1), encode_size = 16, class_num = 10, base_depth = 32):
        
        super(Decoder,self).__init__();
        
        self.input_shape = input_shape;
        self.encode_size = encode_size;
        self.class_num = class_num;
        self.base_depth = base_depth;
        
        self.reshape = tf.keras.layers.Reshape([1, 1, encode_size]);
        self.conv1 = tf.keras.layers.Conv2DTranspose(filters = 2 * base_depth, kernel_size = (7,7), strides = (1,1), padding = 'valid');
        self.bn1 = [tf.keras.layers.BatchNormalization() for i in range(class_num)];
        self.relu1 = tf.keras.layers.LeakyReLU();
        self.conv2 = tf.keras.layers.Conv2DTranspose(filters = 2 * base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same');
        self.bn2 = [tf.keras.layers.BatchNormalization() for i in range(class_num)];
        self.relu2 = tf.keras.layers.LeakyReLU();
        self.conv3 = tf.keras.layers.Conv2DTranspose(filters = 2 * base_depth, kernel_size = (5,5), strides = (2,2), padding = 'same');
        self.bn3 = [tf.keras.layers.BatchNormalization() for i in range(class_num)];
        self.relu3 = tf.keras.layers.LeakyReLU();
        self.conv4 = tf.keras.layers.Conv2DTranspose(filters = base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same');
        self.bn4 = [tf.keras.layers.BatchNormalization() for i in range(class_num)];
        self.relu4 = tf.keras.layers.LeakyReLU();
        self.conv5 = tf.keras.layers.Conv2DTranspose(filters = base_depth, kernel_size = (5,5), strides = (2,2), padding = 'same');
        self.bn5 = [tf.keras.layers.BatchNormalization() for i in range(class_num)];
        self.relu5 = tf.keras.layers.LeakyReLU();
        self.conv6 = tf.keras.layers.Conv2DTranspose(filters = base_depth, kernel_size = (5,5), strides = (1,1), padding = 'same');
        self.bn6 = [tf.keras.layers.BatchNormalization() for i in range(class_num)];
        self.relu6 = tf.keras.layers.LeakyReLU();
        self.conv7 = tf.keras.layers.Conv2D(filters = 1, kernel_size = (5,5), strides = (1,1), padding = 'same');
        self.flatten = tf.keras.layers.Flatten();
        self.bernoulli = tfp.layers.IndependentBernoulli(event_shape = input_shape, convert_to_tensor = tfp.distributions.Bernoulli.logits);
        
    def call(self, input, label):
        
        result = self.reshape(input);
        result = self.conv1(result);
        result = self.bn1[label](result);
        result = self.relu1(result);
        result = self.conv2(result);
        result = self.bn2[label](result);
        result = self.relu2(result);
        result = self.conv3(result);
        result = self.bn3[label](result);
        result = self.relu3(result);
        result = self.conv4(result);
        result = self.bn4[label](result);
        result = self.relu4(result);
        result = self.conv5(result);
        result = self.bn5[label](result);
        result = self.relu5(result);
        result = self.conv6(result);
        result = self.bn6[label](result);
        result = self.relu6(result);
        result = self.conv7(result);
        result = self.flatten(result);
        result = self.bernoulli(result);
        assert isinstance(result, tfp.distributions.Distribution);
        return result;

class CVAE(tf.keras.Model):
    
    def __init__(self, input_shape = (28, 28, 1), encode_size = 16, class_num = 10, base_depth = 32):
        
        super(CVAE, self).__init__();
        
        self.input_shape = input_shape;
        self.encode_size = encode_size;
        self.class_num = class_num;
        self.base_depth = base_depth;
        
        self.encoder = Encoder(encode_size, class_num, base_depth);
        self.decoder = Decoder(input_shape, encode_size, class_num, base_depth);
        self.prior = tfp.distributions.Independent(tfp.distributions.Normal(loc = tf.zeros(encode_size), scale = 1));
    
    def call(self, input, label, weight = 1.0):
        
        code_distr = self.encoder(input, label);
        kl_loss = tfp.layers.KLDivergenceAddLoss(self.prior, weight = 1.0)(code_distr);
        code = code_distr.sample();
        sample_distr = self.decoder(code, label);
        likelihood_loss = -sample_distr.log_prob(input);

        return kl_loss + likelihood_loss;
    
    def sample(self, label, batch_size = 1):
        
        assert 0 <= label < self.class_num;
        code = self.prior.sample(sample_shape = (batch_size));
        sample_distr = self.decoder(code, label);
        return sample_distr.sample();

if __name__ == "__main__":
    
    cvae = CVAE();
