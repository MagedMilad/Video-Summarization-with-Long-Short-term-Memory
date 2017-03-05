import numpy as np
import sys,os
from keras.preprocessing.sequence import pad_sequences
from skimage.util import shape
from operator import itemgetter
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from enum import Enum
import h5py
from datetime import datetime
import random


# dataset path

tvsum = "data/Data_TVSum_google_p5.h5"
summe = "data/Data_SumMe_google_p5.h5"
ovp = "data/Data_OVP_google_p5.h5"
youtube = "data/Data_Youtube_google_p5.h5"

train_x = []
train_y = []
test_x = []
test_y = []

train_shuffled_idx = []
test_shuffled_idx = []


# settings:
#     c - canonical
#     a - augumented
#     t - transfer
# dataset:
#     t - tvsum
#     s - summe


def load_settings(dataset, setting, shuffle=False):
    clear_settings()
    random.seed(datetime.now())
    if setting == "c" and dataset == "s":
        eighty_twenty(summe,shuffle)
    if setting == "c" and dataset == "t":
        eighty_twenty(tvsum,shuffle)

    if setting == "a" and dataset == "s":
        eighty_twenty(summe,shuffle)
        full_training(tvsum)
        full_training(ovp)
        full_training(youtube)
    if setting == "a" and dataset == "t":
        eighty_twenty(tvsum,shuffle)
        full_training(summe)
        full_training(ovp)
        full_training(youtube)

    if setting == "t" and dataset == "s":
        full_testing(summe)
        full_training(tvsum)
        full_training(ovp)
        full_training(youtube)
    if setting == "t" and dataset == "t":
        full_testing(tvsum)
        full_training(summe)
        full_training(ovp)
        full_training(youtube)
    reverse_x_axes(shuffle)


def clear_settings():
    global train_x, train_y, test_x, test_y, train_shuffled_idx, test_shuffled_idx
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    train_shuffled_idx = []
    test_shuffled_idx = []


def eighty_twenty(file_name,shuffle):
    f = h5py.File(file_name, 'r')
    idx = [i + 1 for i in range(50)]
    if file_name == summe:
        idx = [i + 1 for i in range(25)]
    if shuffle:
        random.shuffle(idx)
    train_idx = idx[0: int(0.8 * len(idx))]
    test_idx = idx[int(0.8 * len(idx)): len(idx)]
    for _, i in enumerate(train_idx):
        train_x.append(f["fea_{}".format(i)].value)
        train_y.append(f["gt_1_{}".format(i)].value)
    for _, i in enumerate(test_idx):
        test_x.append(f["fea_{}".format(i)].value)
        test_y.append(f["gt_1_{}".format(i)].value)


def full_training(file_name):
    f = h5py.File(file_name, 'r')
    idx = [i + 1 for i in range(50)]
    if file_name == youtube:
        idx = [i for i in range(11, 51, 1)]
    for _, i in enumerate(idx):
        if (file_name == youtube and i != 22) or (file_name == summe and i < 26):
            train_x.append(f["fea_{}".format(i)].value)
            train_y.append(f["gt_1_{}".format(i)].value)


def full_testing(file_name):
    f = h5py.File(file_name, 'r')
    idx = [i + 1 for i in range(50)]
    for _, i in enumerate(idx):
        if (file_name == summe and i < 26) or (file_name == tvsum):
            test_x.append(f["fea_{}".format(i)].value)
            test_y.append(f["gt_1_{}".format(i)].value)


def reverse_x_axes(shuffle):
    global train_shuffled_idx, test_shuffled_idx
    for i in range(len(train_x)):
        train_x[i] = np.swapaxes(train_x[i], 0, 1)
    for i in range(len(test_x)):
        test_x[i] = np.swapaxes(test_x[i], 0, 1)
    for i in range(len(train_y)):
        if len(train_y[i]) == 1:
            train_y[i] = np.swapaxes(train_y[i], 0, 1)
    for i in range(len(test_y)):
        if len(test_y[i]) == 1:
            test_y[i] = np.swapaxes(test_y[i], 0, 1)
    train_shuffled_idx = [i for i in range(len(train_x))]
    test_shuffled_idx = [i for i in range(len(test_x))]
    if shuffle:
        random.shuffle(train_shuffled_idx)
        random.shuffle(test_shuffled_idx)


def get_train_item(item):
    return np.array(train_x[train_shuffled_idx[item]]), np.array(train_y[train_shuffled_idx[item]])


def get_test_item(item):
    return np.array(test_x[test_shuffled_idx[item]]), np.array(test_y[test_shuffled_idx[item]])


def get_train_size():
    return int(len(train_x)*0.8)


def get_validation_size():
    return int(len(train_x) * 0.2)


def get_test_size():
    return int(len(test_x))


# x_path = "/home/magedmilad/Video-Summarization-with-Long-Short-term-Memory/Feature_Extractor/frames___/"
# y_path = "/home/magedmilad/Video-Summarization-with-Long-Short-term-Memory/Video Sampling/output_sumME/"


# def get_test_sample(idx,dataset):
#     y = []
#     x = []
#     with open(x_path + dataset[20 + idx] + '/out.txt') as textfile:
#         for line in textfile:
#             if line != '\n':
#                 temp = line.split()
#                 x.append([float(x_) for x_ in temp])
#     with open(y_path+dataset[20+idx]) as textfile:
#         for line in textfile:
#             if line != '\n':
#                 temp = line.split()
#                 temp = [float(y_) for y_ in temp]
#                 y.append(temp)
#     x = np.array(x)
#     y = np.array(y)
#     y = np.swapaxes(y,0,1)
#     while x.shape[0] != y.shape[0]:
#         x = np.delete(x,len(y),axis=0)
#     return x,y
#
#
# def get_train_sample(idx,dataset):
#     y = []
#     x = []
#     with open(x_path + dataset[idx] + '/out.txt') as textfile:
#         for line in textfile:
#             if line != '\n':
#                 temp = line.split()
#                 x.append([float(x_) for x_ in temp])
#     with open(y_path+dataset[idx]) as textfile:
#         for line in textfile:
#             if line != '\n':
#                 temp = line.split()
#                 temp = [float(y_) for y_ in temp]
#                 y.append(temp)
#
#     x = np.array(x)
#     y = np.array(y)
#     y = np.swapaxes(y,0,1)
#
#     while x.shape[0] != y.shape[0]:
#         x = np.delete(x,len(y),axis=0)
#     return x,y


# def to_key_frame(y , per = 0.15):
#     weight = []
#     for i in range(y.shape[0]//10):
#         temp = np.average(np.array(y[i*10:i*10+10]))
#         weight.append((temp,i))
#     weight.sort(key=itemgetter(0),reverse=True)
#     slots = y.shape[0]//10*per
#     idx = 0
#     chosen = []
#     while idx < slots:
#         chosen.append(weight[idx][1])
#         idx += 1
#     ret = np.zeros((y.shape[0],y.shape[1]))
#     for i in range(len(chosen)):
#         ret[chosen[i]*10:chosen[i]*10+10][:]=1
#     return np.array(ret)


def get_all_test(batch_size,timesteps,features_size,output_size):

    sequence_stack = []
    for _ in range(batch_size):
        sequence_stack.append([])
    nb_clips_stack = np.zeros(batch_size).astype(np.int64)
    for video_id in range(get_test_size()):
        min_pos = np.argmin(nb_clips_stack)
        sequence_stack[min_pos].append(video_id)
        nb_clips_stack[min_pos] += get_test_item(video_id)[0].shape[0]

    min_sequence = np.min(nb_clips_stack)
    max_sequence = np.max(nb_clips_stack)
    nb_batches_long = max_sequence // timesteps + 1
    nb_batches = min_sequence // timesteps
    print('Number of batches: {}'.format(nb_batches))

    video_features = np.zeros((nb_batches_long * batch_size * timesteps, features_size))
    output = np.zeros((nb_batches_long * batch_size * timesteps, output_size))
    index = np.arange(nb_batches_long * batch_size * timesteps)

    for i in range(batch_size):
        batch_index = index // timesteps % batch_size == i

        pos = 0
        for video_id in sequence_stack[i]:
            x,y = get_test_item(video_id)
            vid_features = x
            nb_instances = vid_features.shape[0]

            output_classes = y

            video_index = index[batch_index][pos:pos + nb_instances]
            video_features[video_index, :] = vid_features
            output[video_index] = output_classes

            pos += nb_instances

    video_features = video_features[:nb_batches * batch_size * timesteps, :]
    video_features = video_features.reshape((nb_batches * batch_size, timesteps, features_size))

    output = output[:nb_batches * batch_size * timesteps, :]
    output = output.reshape((nb_batches * batch_size, timesteps, output_size))

    return video_features,output


def get_all_train(batch_size,timesteps,features_size,output_size):

    sequence_stack = []
    for _ in range(batch_size):
        sequence_stack.append([])
    nb_clips_stack = np.zeros(batch_size).astype(np.int64)

    for video_id in range(get_train_size()):
        min_pos = np.argmin(nb_clips_stack)
        sequence_stack[min_pos].append(video_id)
        nb_clips_stack[min_pos] += get_train_item(video_id)[0].shape[0]

    min_sequence = np.min(nb_clips_stack)
    max_sequence = np.max(nb_clips_stack)
    nb_batches_long = max_sequence // timesteps + 1
    nb_batches = min_sequence // timesteps
    print('Number of batches: {}'.format(nb_batches))

    video_features = np.zeros((nb_batches_long * batch_size * timesteps, features_size))
    output = np.zeros((nb_batches_long * batch_size * timesteps, output_size))
    index = np.arange(nb_batches_long * batch_size * timesteps)

    for i in range(batch_size):
        batch_index = index // timesteps % batch_size == i

        pos = 0
        for video_id in sequence_stack[i]:
            x,y = get_train_item(video_id)
            vid_features = x
            nb_instances = vid_features.shape[0]
            output_classes = y
            video_index = index[batch_index][pos:pos + nb_instances]
            video_features[video_index, :] = vid_features
            output[video_index] = output_classes

            pos += nb_instances

    video_features = video_features[:nb_batches * batch_size * timesteps, :]
    # assert np.all(np.any(video_features, axis=1))
    video_features = video_features.reshape((nb_batches * batch_size, timesteps, features_size))

    output = output[:nb_batches * batch_size * timesteps, :]
    # assert np.all(np.any(output, axis=1))
    output = output.reshape((nb_batches * batch_size, timesteps, output_size))

    return video_features,output

