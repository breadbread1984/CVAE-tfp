# CVAE-tfp

simple conditional variational autoencoder (CVAE) implemented with TFP

## Introduction

conditional variational autoencoder (CVAE) is a generator which can generate certain class of samples according to given class label. the implement of this CVAE adopts conditional instance normalization as a measure to provide condition information. the simplicity of the code attributes to the powerful utilities provided by tensorflow 2.0 and tensorflow probability library.

## how to create mnist dataset

run the following command to create mnist dataset in tfrecord format.

```Bash
python3 create_dataset.py
```

## how to train

launch the training by executing

```Bash
python3 train_CVAE.py
```

you can supervise the training by tensorboard with the command

```Bash
tensorboard --logdir checkpoints
```

the changing of loss and the evolving of the generated hand writing figure images are shown on tensorboard.

## how to test

sample hand writing figure images by executing

```Bash
python3 CVAE.py
```

10 digits will be sampled and shown.

