#!/usr/bin/python
# -*- coding: utf-8 -*-

import os;
import numpy as np;
import tensorflow as tf;
import tensorflow_datasets as tfds;

def create_mnist():

    # load dataset
    mnist_builder = tfds.builder("mnist");
    mnist_builder.download_and_prepare();
    # try to load the dataset once
    mnist_train = tfds.load(name = "mnist", split = tfds.Split.TRAIN, download = False);
    mnist_test = tfds.load(name = "mnist", split = tfds.Split.TEST, download = False);

if __name__ == "__main__":

    create_mnist();

