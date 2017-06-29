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


# dataset = ['-esJrBWj2d8', '0tmA_C6XwfM', '37rzWOQsNIw', '3eYKfiOEJNs', '4wU_LUjG5Ic', '91IHQYk1IQM', '98MoyGZKHXc', 'AwmHb44_ouw', 'Bhxk-O1Y7Ho', 'E11zDS9XGzg', 'EE-bNr36nyA', 'EYqVtI9YWJA', 'GsAD1KT1xo8', 'HT5vyqe0Xaw', 'Hl-__g2gn_A', 'J0nA4VgnoCo', 'JKpqYvAdIsw', 'JgHubY5Vw3Y', 'LRw_obCPUt0', 'NyBmCxDoHJU', 'PJrm840pAUI', 'RBCABdttQmI', 'Se3oxnaPsz0', 'VuWGsYPqAX8', 'WG0MBPpPC6I', 'WxtbjNsCQ8A', 'XkqCExn6_Us', 'XzYM3PfTM4w', 'Yi4Ij2NM7U4', '_xMr-HKMfVA', 'akI8YFjEmUw', 'b626MiF1ew4', 'byxOvuiIJV0', 'cjibtmSLxQ4', 'eQu1rNs0an0', 'fWutDQy1nnY', 'gzDbaEs1Rlg', 'i3wAGJaaktw', 'iVt07TCkFM0', 'jcoYJXDG9sw', 'kLxoNp-UchI', 'oDXZc0tZe04', 'qqR6AEXwxoQ', 'sTEELN-vY30', 'uGu_10sucQo', 'vdmoEJ5YbrQ', 'xmEERLqJ2kU', 'xwqBXPGE9pQ', 'xxdtq8mxegs', 'z_6gVvQb2d0']

# dataset = ['Air_Force_One', 'Base_jumping', 'Bearpark_climbing', 'Bike_Polo', 'Bus_in_Rock_Tunnel', 'Car_railcrossing', 'Cockpit_Landing', 'Cooking', 'Eiffel_Tower', 'Excavators_river_crossing', 'Fire_Domino', 'Jumps', 'Kids_playing_in_leaves', 'Notre_Dame', 'Paintball', 'Playing_on_water_slide', 'Saving_dolphines', 'Scuba', 'St_Maarten_Landing', 'Statue_of_Liberty', 'Uncut_Evening_Flight', 'Valparaiso_Downhill', 'car_over_camera', 'paluma_jump', 'playing_ball']


def eval(dataset,setting):
    x = Input(batch_shape=(1, 1, features_size,), name='x')
    # norm = BatchNormalization(name='norm')(x)

    lstmR = LSTM(256, return_sequences=True, name='lstmR', stateful=True)(x)
    lstmL = LSTM(256, return_sequences=True, go_backwards=True, name='lstmL', stateful=True)(x)

    m = merge([x, lstmR, lstmL], mode='concat', name='merge')

    dense = TimeDistributed(Dense(256, activation='sigmoid', name='dense'))(m)
    y = TimeDistributed(Dense(1, activation='sigmoid', name='y'))(dense)

    model = Model(input=x, output=y)

    # model.summary()
    model.load_weights("{}.h5".format(get_model_name(dataset,setting)))

    # todo change loss and optimizer
    model.compile(loss='mean_squared_error',
                  optimizer=SGD(),
                  metrics=['accuracy'])

    # plot(model, to_file='model.png', show_shapes=True)



    print('eval...')
    mean = 0
    for i in range(get_validation_size()):
        x,y = get_train_item(i+get_train_size())
        pred_y = model.predict(np.expand_dims(x, axis=1),batch_size=1,verbose=2)
        model.reset_states()
        pred_y = np.array(pred_y)
        pred_y = pred_y.reshape(y.shape[0],y.shape[1])
        # pred_y = normalize(pred_y, axis=0)
        pred_y = to_keyshots_feature(x,pred_y)
        y = to_keyshots_feature(x,y)
        # f = test_f_score(dataset[i+offset], pred_y)
        f=f1_score(y, pred_y) * 100
        mean+=f
        # print('f-socre = {} => {}'.format(f1_score(y, pred_y) * 100,f))
    print (mean/(get_validation_size()))
    return (mean/(get_validation_size()))


# def test_f_score(video_name,pred_y, HOMEMAT='/home/magedmilad/Video-Summarization-with-Long-Short-term-Memory/Video Sampling/mat_',HOMEDATA='/home/magedmilad/Video-Summarization-with-Long-Short-term-Memory/Video Sampling/vids_'):
#     extended_pred_y = get_summery(HOMEDATA+'/'+video_name+'.webm',pred_y)
#     f_score = evaluateSummary(extended_pred_y,video_name,HOMEMAT)
#     # print (f_score)
#     return f_score


# if __name__ == "__main__":
    # random.seed(datetime.now())
    # random.shuffle(dataset)
    # predect(dataset)
