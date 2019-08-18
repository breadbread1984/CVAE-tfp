#!/usr/bin/python3

import os;
import tensorflow as tf;
import tensorflow_datasets as tfds;
from CVAE import CVAE;

batch_size = 256;

def parse_function(feature):
    data = feature["image"];
    data = tf.cast(data,dtype = tf.float32) / 255.;
    label = tf.cast(feature["label"], dtype = tf.int32);
    return (data, label), data;

def main():
    
    cvae = CVAE();
    #load dataset
    trainset = tfds.load(name = "mnist", split = tfds.Split.TRAIN, download = False);
    trainset = trainset.map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);
    testset = tfds.load(name = "mnist", split = tfds.Split.TRAIN, download = False);
    testset = testset.map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE);

    cvae.compile(optimizer = tf.keras.optimizers.Adam(1e-3), loss = lambda x, sample: -sample.log_prob(x));
    cvae.fit(trainset, epochs = 15, validation_data = testset);
    cvae.save_weights('cvae.h5');

if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    main();

