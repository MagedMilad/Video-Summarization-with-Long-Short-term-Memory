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
from sklearn.model_selection import train_test_split
from predect import predect
from datetime import datetime
import random
from eval import eval
from utils import *

def train(batch_size,dataset,setting,timesteps=10):


    features_size=1024
    output_size=1
    # lr = 1e-5

    x = Input(batch_shape=(batch_size,timesteps, features_size,), name='x')
    lstmR = LSTM(256, return_sequences=True, name='lstmR', stateful=True)(x)
    lstmL = LSTM(256, return_sequences=True, go_backwards=True, name='lstmL', stateful=True)(x)

    m = merge([x, lstmR, lstmL], mode='concat', name='merge')

    dense = TimeDistributed(Dense(256, activation='sigmoid', name='dense'))(m)
    y = TimeDistributed(Dense(1, activation='sigmoid', name='y'))(dense)

    model = Model(input=x, output=y)

    model.summary()

    # todo change loss and optimizer
    model.compile(loss='mean_squared_error',
                  optimizer=SGD(),
                  metrics=['accuracy'])

    plot(model, to_file='model.png', show_shapes=True)

    print("done")

    print('Train...')

    X,Y = get_all_train(batch_size,timesteps,features_size,output_size)

    f_scores = []
    i = 0

    while True:
        i +=1
        print ('Epoch {}'.format(i))
        model.fit(X,
                  Y,
                  batch_size=batch_size,
                  verbose=1,
                  nb_epoch=1,
                  shuffle=False)
        model.reset_states()

        model.save_weights("{}.h5".format(get_model_name(dataset,setting)))

        f_scores.append(eval(dataset,setting))

        if len(f_scores) >=5 and cmp(f_scores,sorted(f_scores, reverse=True)) == 0:
            print (f_scores)
            print ('overfit...')
            break

        if len(f_scores) >=5:
            print (f_scores)
            f_scores.remove(f_scores[0])

    model_json = model.to_json()
    with open("{}.json".format(get_model_name(dataset,setting)), "w") as json_file:
        json_file.write(model_json)
    model.save_weights("{}.h5".format(get_model_name(dataset,setting)))
    print("Saved model to disk")

    predect(dataset,setting)



