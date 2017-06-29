from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD, Adam
from keras.utils.visualize_util import plot
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional, Input, merge,BatchNormalization
from keras.models import model_from_json
from preprocess import *
from keras import metrics
from sklearn.metrics import f1_score
from keras.layers.wrappers import TimeDistributed
from datetime import datetime
import random
from keyshots import to_keyshots_feature
from utils import get_summery
from sumMeEval import evaluateSummary
from sklearn.preprocessing import normalize
from utils import *

features_size = 1024

def predect(dataset,setting,fea,dir):
    x = Input(batch_shape=(1, 1, features_size,), name='x')
    # norm = BatchNormalization(name='norm')(x)

    lstmR = LSTM(256, return_sequences=True, name='lstmR', stateful=True)(x)
    lstmL = LSTM(256, return_sequences=True, go_backwards=True, name='lstmL', stateful=True)(x)

    m = merge([x, lstmR, lstmL], mode='concat', name='merge')

    dense = TimeDistributed(Dense(256, activation='sigmoid', name='dense'))(m)
    y = TimeDistributed(Dense(1, activation='sigmoid', name='y'))(dense)

    model = Model(input=x, output=y)

    model.summary()
    # print (get_model_name(dataset,setting))
    model.load_weights("/home/magedmilad/PycharmProjects/VSA/vsLSTM/dataset: TVSum, Setting: Transfer.h5")

    # todo change loss and optimizer
    model.compile(loss='mean_squared_error',
                  optimizer=SGD(),
                  metrics=['accuracy'])

    # plot(model, to_file='model.png', show_shapes=True)



    print('summarize...')
    # mean = 0
    # for i in range(get_test_size()):
    #     x, y = get_test_item(i)

    Y = model.predict(np.expand_dims(fea, axis=1),batch_size=1,verbose=2)
    # model.reset_states()
    Y = np.array(Y)
    fea = np.array(fea)
    Y = Y.reshape(fea.shape[0],1)
    Y = to_keyshots_feature(fea,Y)

    return get_summery(dir,Y)


    # y = to_keyshots_feature(x,y)
    # mean+=(f1_score(y,pred_y)*100)
    # print ('f-socre = {}'.format(f1_score(y,pred_y)*100))

    # print (mean/get_test_size())

