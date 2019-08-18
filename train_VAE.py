#!/usr/bin/python3

import os;
import tensorflow as tf;
import tensorflow_datasets as tfds;
from VAE import VAE;

batch_size = 256;

def parse_function(feature):
    data = feature["image"];
    data = tf.cast(data,dtype = tf.float32) / 255.;
    label = feature["label"];
    return data, data;

def main():
    
    vae = VAE();
    #load dataset
    trainset = tfds.load(name = "mnist", split = tfds.Split.TRAIN, download = False);
    trainset = trainset.map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
    testset = tfds.load(name = "mnist", split = tfds.Split.TRAIN, download = False);
    testset = testset.map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);

    vae.compile(optimizer = tf.keras.optimizers.Adam(1e-3), loss = lambda x, sample: -sample.log_prob(x));
    vae.fit(trainset, epochs = 15, validation_data = testset);
    vae.save_weights('vae.h5');

if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    main();

