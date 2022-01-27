# -*- coding: utf-8 -*-
"""
Collecting predictability classes in five categories
"""
import os
import numpy as np
import pandas as pd

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Bidirectional
from keras.layers import Input, Dense,LSTM

import tensorflow
from tensorflow.keras.optimizers import Adam

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ''

def train_test_split_week(week_test, updating = False):
    df_indxs = ttab
    ind_test = df_indxs[(df_indxs['last_week'] >= week_test) & (df_indxs['last_week'] < week_train)]
    ind_test_2 = df_indxs[(df_indxs['last_week'] >= week_train) & (df_indxs['last_week'] < week_test_2)]

    if updating:
        ind_train = df_indxs[(df_indxs['last_week'] >= week_upd) & (df_indxs['last_week'] < week_test)]
    else:
        ind_train = df_indxs[(df_indxs['last_week'] >= week_test-L_win) & (df_indxs['last_week'] < week_test)] #!!!

    return ind_test, ind_train, ind_test_2

class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, df, indexes, L_win, NCATS, batch_size):
        self.data = df
        self.batch_size = batch_size
        self.ind = indexes
        self.L_win = L_win
        self.NCATS = NCATS
    
    def __len__(self):
        return int(np.floor(len(self.ind) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_ind = self.ind[idx * self.batch_size:(idx + 1) * self.batch_size]
        Ck = batch_ind[:, 0]
        month = batch_ind[:, 1] - 1
        ind_x = batch_ind[:, -(self.L_win + 1) : -1]
        ind_y = batch_ind[:, -1]

        X = self.data[ind_x, : ]
        Y = self.data[ind_y, :]
        Y = np.where(self.data[ind_y,:], 1, 0)
        X = X.reshape(self.batch_size, self.L_win, self.NCATS)
        Y = Y.reshape(self.batch_size, self.NCATS) 
        return [X, Ck, month], Y

def create_model():
    inp = Input(shape=(L_win, NCATS))
    inp_ck = Input(shape = (1, ))
    inp_m = Input(shape = (1, ))

    lay = LSTM(NFILTERS, return_sequences = True)(inp)
    lay2 = LSTM(NFILTERS)(lay)
    trg_clf = Dense(NCATS, activation = 'sigmoid')(lay2)

    model_clf = Model(inputs = [inp, inp_ck, inp_m], outputs = trg_clf)
    model_clf.compile(loss = 'binary_crossentropy', optimizer = OPTIM, metrics = ['accuracy']) 

    return model_clf   

def answers_for_classes(classA, classB):
    ind_test_A = ind_test[ind_test['id'].isin(classA)]
    ind_test_B = ind_test[ind_test['id'].isin(classB)]
    y_pred_A, y_true_A = get_answers_for_classes(ind_test_A)
    y_pred_B, y_true_B = get_answers_for_classes(ind_test_B)
    ind_cat = table.columns.get_loc(pred_cat) - 2
    return y_pred_A[:, ind_cat], y_true_A[:, ind_cat], y_pred_B[:, ind_cat], y_true_B[:, ind_cat]

def get_answers_for_classes(ind_test):
    model_RNN = create_model()
    model_RNN.load_weights("LSTM.h5")
    g_test = DataGenerator(table.values[:,2:], ind_test.values, L_win, NCATS, BATCH_SIZE)
    y_pred = model_RNN.predict(g_test)
    y_true = np.vstack([g_test[i][1] for i in range(len(g_test))])
    del model_RNN
    return y_pred, y_true

def train_and_predict(updating = False):
    g_train = DataGenerator(table.values[:, 2:], ind_train.values, L_win, NCATS, BATCH_SIZE)

    g_test = DataGenerator(table.values[:, 2:], ind_test.values, L_win, NCATS, BATCH_SIZE)
    g_test2 = DataGenerator(table.values[:, 2:], ind_test_2.values, L_win, NCATS, BATCH_SIZE)
  
    model_RNN = create_model()
    if updating:
        model_RNN.load_weights("LSTM.h5")
    
    if step==0:
        model_RNN.fit(g_train, validation_data = g_test, epochs = NB_EPOCH, verbose = 0)
        model_RNN.save_weights('LSTM.h5')

    y_pred = model_RNN.predict(g_test)
    y_pred2 = model_RNN.predict(g_test2)
    y_true = np.vstack([g_test[i][1] for i in range(len(g_test))])
    ind_cat = table.columns.get_loc(pred_cat) - 2
    #model_RNN.save_weights('LSTM.h5')
    del model_RNN
    return y_pred[:, ind_cat], y_true[:, ind_cat], y_pred2[:, ind_cat]

def count_coefficient(y_pred, y_true, y_pred2):

    ind_test_loc = ind_test.copy()
    ind_test_loc2 = ind_test_2.copy()

    #the first interval
    ind_test_loc['target'] = table.values[ind_test_loc.values[:, -1], table.columns.get_loc(pred_cat)]
    ind_test_loc['target'] = np.where(ind_test_loc['target'], 1, 0)
    ind_test_loc['predicted_prob'] = np.zeros(len(ind_test_loc))
    ind_test_loc.iloc[:len(y_pred), ind_test_loc.columns.get_loc('predicted_prob')] = y_pred
    ind_test_loc = ind_test_loc[:len(y_pred)]
    ind_test_loc['num'] = abs(ind_test_loc['target'] - ind_test_loc['predicted_prob'])
    num = ind_test_loc.groupby('id')['num'].sum()
    den_n = ind_test_loc.groupby('id')['target'].count()
    CE = num / den_n
    CE = 1 - CE

    #the second interval
    ind_test_loc2['target'] = table.values[ind_test_loc2.values[:, -1], table.columns.get_loc(pred_cat)]
    ind_test_loc2['target'] = np.where(ind_test_loc2['target'], 1, 0)
    ind_test_loc2['predicted_prob'] = np.zeros(len(ind_test_loc2))
    ind_test_loc2.iloc[:len(y_pred2), ind_test_loc2.columns.get_loc('predicted_prob')] = y_pred2
    ind_test_loc2 = ind_test_loc2[:len(y_pred2)]
    ind_test_loc2['num'] = abs(ind_test_loc2['target'] - ind_test_loc2['predicted_prob'])
    num2 = ind_test_loc2.groupby('id')['num'].sum()
    den_n2 = ind_test_loc2.groupby('id')['target'].count()
    CE2 = num2 / den_n2
    CE2 = 1 - CE2
    return CE, CE2

def predictability_classes(CE):
    A=np.where(CE>np.median(CE))[0] 
    B=np.where(CE<np.median(CE))[0]
    return A, B

def train_test_split_for_class_identification():

    #CE, CE2 = count_coefficient()
    ind = np.arange(week_test, week_train)
    ind_test = np.arange(week_train, week_test_2)

    X_train = table[table['WEEK'].isin(ind)]
    X_train = X_train[X_train['id'].isin(CE.keys())]
    X_test = table[table['WEEK'].isin(ind_test)]
    X_test = X_test[X_test['id'].isin(CE2.keys())]

    test_ids = np.unique(X_test['id'])

    return X_train, X_test, test_ids

def reshape_train_and_test(updating = False):
    list_X, list_X_test = [], []
    X_train, X_test, test_ids = train_test_split_for_class_identification()

    un_ids = np.unique(X_train['id'])
    Y = np.zeros(len(un_ids))
    un_ids_test = np.unique(X_test['id'])
    for c_id in un_ids:
        cur = X_train[X_train['id'] == c_id][pred_cat].values
        list_X.append(cur)
    for i in range(len(un_ids)):
        if un_ids[i] in A:
            Y[i] = 0
        elif un_ids[i] in B:
            Y[i] = 1
    for i in range(len(list_X)):
        list_X[i] = list_X[i][-parameter:]
    X = np.vstack(list_X)
    X = X.reshape(len(list_X), n_timesteps, 1)
    Y = pd.get_dummies(Y).values
    Y = Y.reshape(len(list_X), 2)
    for c_id in un_ids_test:
        cur = X_test[X_test['id'] == c_id][pred_cat].values
        list_X_test.append(cur)
    
    for i in range(len(list_X_test)):
        list_X_test[i] = list_X_test[i][-parameter:]
    
    X_test_n = np.vstack(list_X_test)
    X_test_n = X_test_n.reshape(len(list_X_test), n_timesteps, 1)

    return X, Y, X_test_n, test_ids

def BiLSTM_model(updating = False):
    X, Y, X_test_n, test_ids = reshape_train_and_test(updating = updating)
    model = Sequential()
    model.add(Bidirectional(LSTM(BiNFILTERS, input_shape=(n_timesteps, 1), 
                                 return_sequences = False)))
    model.add(Dense(2, activation = 'softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['acc'])

    X = X.astype('float32')
    Y = Y.astype('float32')
    X_test_n = X_test_n.astype('float32')

    if updating: # predicting with once calculated weights
        model(np.zeros((1,*X.shape[1:])))
        model.load_weights('BiLSTM.h5')

    if step==0: # Model gets fitting only once
        model.fit(X, Y, epochs = BiNB_EPOCHS, batch_size = BiBATCH_SIZE, verbose=0)
        model.save_weights('BiLSTM.h5')

    yhat = model.predict(X_test_n).argmax(1)

    pred_A, pred_B = [], []
    for i in range(len(yhat)):
        if (yhat[i] == 0):
            pred_A.append(test_ids[i])
        else:
            pred_B.append(test_ids[i])
    del model
    return pred_A, pred_B

L_win = 4 
delta = 2

n_timesteps = 4
parameter=4
num_month_client=3

NFILTERS = 64
lr = 0.001
BATCH_SIZE = 64
NB_EPOCH = 10

BiNFILTERS = 20
BiNB_EPOCHS = 30
BiBATCH_SIZE = 32

OPTIM = Adam(learning_rate=lr)
datadir='data/'
setname='D1'

table=pd.read_csv(datadir+setname+'table.csv')
print('Loaded data from '+datadir+setname+'table.csv' )
ttab=pd.read_csv(datadir+setname+'indtab.csv').astype(int)
print('Loaded indices from '+datadir+setname+'indtab.csv' )
week_test = 6 - delta
week_train = 10 - delta
week_test_2 = 14 - delta
NFILTERS = 64
BATCH_SIZE = 64
NCATS = table.shape[1] - 2
colors = ['k','g','r','g', 'r']
linestyles = ['-', '-', '-', '-', '-']
accA, accB, accTot=[],[],[]
ce=[]
n_cats=['63','60','51','29','13']
total=pd.DataFrame(columns=['step','id','cat63','cat60','cat51','cat29','cat13'])
for step in range(20):
    res=pd.DataFrame(columns=['step','id','cat63','cat60','cat51','cat29','cat13'])
    week_test += delta
    week_train += delta
    week_test_2 += delta
    week_upd = week_test - delta
    if week_test_2>ttab.last_week.max():
        break
    for pred_cat in n_cats:
        clmn='cat'+pred_cat
        K.clear_session()
        print('step # %i ===category: %s; week_test:%i; week_train:%i;  week_test_2:%i; week_upd:%i; ========='%(step, 
                                                clmn, week_test, week_train, week_test_2, week_upd))
        if step==0:
            upd=False
        else:
            upd=True

        ind_test, ind_train, ind_test_2 = train_test_split_week(week_test, updating = upd)
        y_pred, y_true, y_pred2 = train_and_predict(updating = upd)
        CE, CE2 = count_coefficient(y_pred, y_true, y_pred2)
        A, B = predictability_classes(CE)
        pred_A, pred_B = BiLSTM_model(updating = upd)
        
        res.step=['step'+str(step).zfill(2)]*(len(pred_A)+len(pred_B))
        res.id=np.concatenate((pred_A,pred_B))
        res[clmn]=np.concatenate((np.ones(len(pred_A)), np.zeros(len(pred_B)))).astype(int)
    total=pd.concat((total,res))
total=total.sort_values(by=['id', 'step']).reset_index().drop('index',axis=1)
total.to_csv(datadir+setname+'total.csv', index=False)
print('Results saved in '+datadir+setname+'total.csv' )