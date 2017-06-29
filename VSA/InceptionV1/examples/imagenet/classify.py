#!/usr/bin/env python
import argparse
import numpy as np
import tensorflow as tf
import os.path as osp

import models
import dataset
import os, sys
import time

video_feature = {}

def display_results(image_paths, probs):
    '''Displays the classification results given the class probability for each image'''
    # Get a list of ImageNet class labels
    with open('imagenet-classes.txt', 'rb') as infile:
        class_labels = map(str.strip, infile.readlines())
    # Pick the class with the highest confidence for each image
    class_indices = np.argmax(probs, axis=1)
    # Display the results
    print('\n{:20} {:30} {}'.format('Image', 'Classified As', 'Confidence'))
    print('-' * 70)
    for img_idx, image_path in enumerate(image_paths):
        img_name = osp.basename(image_path)
        class_name = class_labels[class_indices[img_idx]]
        confidence = round(probs[img_idx, class_indices[img_idx]] * 100, 2)
        print('{:20} {:30} {} %'.format(img_name, class_name, confidence))


def classify(model_data_path, image_paths):
    '''Classify the given images using GoogleNet.'''

    # Get the data specifications for the GoogleNet model
    spec = models.get_data_spec(model_class=models.GoogleNet)

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32,
                                shape=(None, spec.crop_size, spec.crop_size, spec.channels))

    # Construct the network
    net = models.GoogleNet({'data': input_node})

    # Create an image producer (loads and processes images in parallel)
    image_producer = dataset.ImageProducer(image_paths=image_paths, data_spec=spec)

    with tf.Session() as sesh:
        # Start the image processing workers
        coordinator = tf.train.Coordinator()
        threads = image_producer.start(session=sesh, coordinator=coordinator)

        # Load the converted parameters
        print('Loading the model')
        net.load(model_data_path, sesh)

        # Load the input image
        print('Loading the images')
        indices, input_images = image_producer.get(sesh)

        # Perform a forward pass through the network to get the class probabilities
        print('Classifying')
        # probs = sesh.run(net.get_output(), feed_dict={input_node: input_images})
        # display_results([image_paths[i] for i in indices], probs)



        # # ////////////////////////////////////////////////////////////////////////////////////
        feature_tensor = sesh.graph.get_tensor_by_name('pool5_7x7_s1:0')
        features = sesh.run(feature_tensor, feed_dict={input_node: input_images})
        features = np.squeeze(features)

        for i,j in enumerate(indices):
            video_feature[image_paths[j]] = features[i]

        # print features.shape
        # print features
        # ////////////////////////////////////////////////////////////////////////////////////

        # Stop the worker threads
        coordinator.request_stop()
        coordinator.join(threads, stop_grace_period_secs=2)

def conv(folder):
    if folder.endswith(".txt"):
        return -1
    return int(folder[5:-4])

def sortKey(folder):
    folder = osp.basename(folder)
    if folder.endswith(".txt"):
        return -1
    return int(folder[5:-4])

def main(dir):
    images = [];
    # dir = "/home/magedmilad/Video-Summarization-with-Long-Short-term-Memory/Video Sampling/frames_/Air_Force_One/"
    # dir = "/home/magedmilad/PycharmProjects/GoogleNet/images/"
    for image_path in sorted(os.listdir(dir), key=conv):
        if image_path.endswith(".txt"):
            continue
        images.append(dir + image_path)
        # video_feature.append({dir + image_path:[]})

    # print images
    # print video_feature

    # for i in range(len(images)/200):
    #     print len(images[i*200:min(len(images),i*200+200)])
    #     classify("/home/magedmilad/PycharmProjects/GoogleNet/data",images[i*200:min(len(images),i*200+200)])

    # now = time.time()
    # i = 0
    # while True:
    #     if(i*200 < len(images)):
    #         classify("/home/magedmilad/PycharmProjects/GoogleNet/data",
    #                  images[i * 200:min(len(images), i * 200 + 200)])
    #         tf.reset_default_graph()
    #         i+=1
    #     else:
    #         break;
    # print (time.time() - now)

    # now = time.time()

    if len(images) / 200 > 0:
        classify("/home/magedmilad/PycharmProjects/VSA/InceptionV1/data", images[0: (len(images) / 200) * 200])
        if len(images) % 200 != 0:
            tf.reset_default_graph()
            classify("/home/magedmilad/PycharmProjects/VSA/InceptionV1/data",
                     images[(len(images) / 200) * 200:len(images)])
    else:
        classify("/home/magedmilad/PycharmProjects/VSA/InceptionV1/data", images[0:len(images)])

    # print (time.time() - now)

    # print video_feature
    # print len(video_feature)

    out = []

    for key in sorted(video_feature.keys(), key=sortKey):
        # print key
        out.append(video_feature[key])

    return  out

def feature_extract(dir):
    return main(dir)

if __name__ == '__main__':
    dir = "/home/magedmilad/Video-Summarization-with-Long-Short-term-Memory/Video Sampling/frames_/Air_Force_One/"
    main(dir)
