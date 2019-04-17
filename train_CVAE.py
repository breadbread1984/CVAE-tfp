#!/usr/bin/python3

import os;
import tensorflow as tf;
import tensorflow_datasets as tfds;
import tensorflow_probability as tfp;
from CVAE import CVAE;

batch_size = 100;

def parse_function(feature):
    data = feature["image"];
    data = tf.cast(data,dtype = tf.float32) / 255.;
    data = tf.random.uniform(tf.shape(data)) > data; # data augmentation
    label = feature["label"];
    return data,label;

def main():
    
    cvae = CVAE(class_num = 10);
    optimizer = tf.keras.optimizers.Adam(1e-3);
    #load dataset
    trainset = tfds.load(name = "mnist", split = tfds.Split.TRAIN, download = False);
    trainset = trainset.map(parse_function).shuffle(batch_size).batch(batch_size);
    testset = tfds.load(name = "mnist", split = tfds.Split.TRAIN, download = False);
    testset = testset.map(parse_function).shuffle(batch_size).batch(batch_size);
    #restore from existing checkpoint
    if False == os.path.exists('cvae_checkpoints'): os.mkdir('cvae_checkpoints');
    checkpoint = tf.train.Checkpoint(model = cvae, optimizer = optimizer, optimizer_step = optimizer.iterations);
    checkpoint.restore(tf.train.latest_checkpoint('cvae_checkpoints'));
    #create log
    log = tf.summary.create_file_writer('cvae_checkpoints');
    #train model
    print('training');
    avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
    while True:
        for (images, labels) in trainset:
            with tf.GradientTape() as tape:
                loss = cvae(images, labels);
                avg_loss.update_state(loss);
            #write log
            if tf.equal(optimizer.iterations % 100, 0):
                with log.as_default():
                    tf.summary.scalar('loss',avg_loss.result(), step = optimizer.iterations);
                    for i in range(10):
                        tf.summary.image(str(i),cvae.sample(labels = i), step = optimizer.iterations);
                print('Step #%d Loss: %.6f' % (optimizer.iterations, avg_loss.result()));
                avg_loss.reset_states();
            grads = tape.gradient(loss, cvae.trainable_variables);
            optimizer.apply_gradients(zip(grads, cvae.trainable_variables));
        #save check point
        checkpoint.save(os.path.join('cvae_checkpoints','ckpt'));

if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    main();
