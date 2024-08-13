#!/usr/bin/env python

from __future__ import print_function

import time
start_time = time.time()

import tensorflow as tf
from keras.models import load_model
from tensorflow import keras
from keras import layers
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras import regularizers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.svm import SVC
from sklearn.utils import class_weight

from matplotlib import pyplot
#import ROOT
import numpy as np
import pandas as pd
import uproot as uprt
import seaborn as sns
import h5py
import csv

from my_functions import compare_train_test_binary
from my_functions import selection_criteria
from my_functions import plot_input_features
from my_functions import AUC_ROC

from keras.callbacks import LambdaCallback

class MyCustomCallback(tf.keras.callbacks.Callback):

  def on_train_batch_begin(self, batch, logs=None):
    print(self.model.layers[0].get_weights())

  def on_train_batch_end(self, batch, logs=None):
    print(self.model.layers[0].get_weights())

def prepare_data():

    # Get data from root files
    #samples = ['output_ZZ', 'out_2000']  ## eventually change this to 'output_ZZ_even', 'out_2000_even' 
    #samples = ['output_ZZ_odd', 'out_2000_odd']
    samples = ["ZZ4l","sample"]
    DataFrames = {} # define empty dictionary to hold dataframes
    Selection = {}    

    Selection_inputs = [] ## don't need
 
    ML_inputs = ["radion_pt", "vis_mass", "vis_mass2","radion_eta", "higgs2_dr", "higgs1_dr", "dphi_H1",
                 "dphi_H1_MET", "dphi_H2", "dphi_H2_MET", "dr_HH", "dr_H1_Rad", "dphi_HH", "dr_H2_Rad", "dphi_rad_MET", 
                 "H2OS", "H1OS", "numBJet"]  

    print('Preparing data')

    #Read training data from parquet files
    for s in samples:
        #file = pd.read_parquet(s + ".paquet",engine="pyarrow")
        DataFrames[s] = pd.read_parquet(s + ".paquet",engine="pyarrow")
        DataFrames[s] = pd.Series({var : DataFrames[s][var].to_numpy() for var in ML_inputs}) #Store everything as a pandas series
        DataFrames[s] = DataFrames[s].iloc[0:8500]

    
    #Old stuff
    #for s in samples: # loop over samples
    #    print(s)
    #    file = uprt.open("/afs/hep.wisc.edu/home/kpoppen/HH4tau/HH4T_/training_samples/" + s + ".root") 

        ## THIS HAS WORKED: file = uprt.open("/afs/hep.wisc.edu/home/kpoppen/HH4tau/HH4T_/outputs/" + s + ".root") 
    #    tree = file['tree_4tau']
    #    DataFrames[s] = tree.arrays(ML_inputs,library="pd")
    #    Selection[s] = tree.arrays(Selection_inputs,library="pd")
        #DataFrames[s] = tree.pd.df(ML_inputs)
        #Selection[s] = tree.pd.df(Selection_inputs)
        #DataFrames[s] = DataFrames[s][ np.vectorize(selection_criteria)(Selection[s].MEM,Selection[s].N_btags_Medium,Selection[s].njets,Selection[s].mbb) ]
    #    DataFrames[s] = DataFrames[s].iloc[0:8500] # first rows of dataframe

    print(DataFrames)

    all_MC = [] # define empty list that will contain all features for the MC
    all_y = [] # define empty list that will contain labels whether an event in signal or background

    for s in samples: # loop over the different samples
        print(s)
        if s!='data': # only MC should pass this
            all_MC.append(DataFrames[s][ML_inputs]) # append the MC dataframe to the list containing all MC features
            if 'out_2000_odd' in s: # only signal MC should pass this
            ## sort between even and odd 

                all_y.append(np.ones(DataFrames[s].shape[0])) # signal events are labelled with 1
            else:
                all_y.append(np.full(DataFrames[s].shape[0],0)) # All backgrounds labelled with 0

    X = np.concatenate(all_MC) # concatenate the list of MC dataframes into a single 2D array of features, called X
    y = np.concatenate(all_y) # concatenate the list of lables into a single 1D array of labels, called y

    
    # make train and test sets
    print('Preparing train and test data')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=492 ) # set the random seed for reproducibility

    scaler = StandardScaler() # initialise StandardScaler
    scaler.fit(X_train) # Fit only to the training data
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_scaled = scaler.transform(X)

    for x,var in enumerate(ML_inputs):
        plot_input_features(X, y, x,var)

    # Save .csv with normalizations
    mean = scaler.mean_
    std  = scaler.scale_
    with open('variable_norm.csv', mode='w') as norm_file:
        headerList = ['', 'mu', 'std']
        norm_writer = csv.DictWriter(norm_file,delimiter=',',fieldnames=headerList) 
        norm_writer.writeheader()
        for x,var in enumerate(ML_inputs): 
            print(var,mean[x],std[x])
            norm_writer.writerow({'': var, 'mu': mean[x], 'std': std[x]})

    X_valid_scaled, X_train_nn_scaled = X_train_scaled[:1000], X_train_scaled[1000:] # first 1000 events for validation
    y_valid, y_train_nn = y_train[:1000], y_train[1000:] # first 1000 events for validation

    print('Input feature correlation')
    print(DataFrames['out_2000_odd'].corr()) #Pearson
    fig = pyplot.figure(figsize=(20, 16))
    corrMatrix = DataFrames['out_2000_odd'].corr()
    ax = pyplot.gca()    
    pyplot.text(0.5, 1.05, "CMS Simulation (Work In Progress)      (13 TeV)", fontweight="bold", horizontalalignment='center',verticalalignment='center', transform=ax.transAxes, fontsize=28)   
    sns.heatmap(corrMatrix, annot=True, cmap=pyplot.cm.Blues)
    pyplot.savefig('correlation.png')

    return X_train_nn_scaled, y_train_nn, X_test_scaled, y_test, X_valid_scaled, y_valid

def nn_model():

    # create model
    model = Sequential()
    ###change dimension here (input_dim = number of variables in ML inputs) 
    model.add(Dense(249, input_dim=18,kernel_regularizer=regularizers.l1_l2(l1=1e-5,l2=7*1e-4)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.09))
    model.add(Dense(24,kernel_regularizer=regularizers.l1_l2(l1=1e-5,l2=7*1e-4)))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.09))
    model.add(Dense(1, activation='sigmoid'))

    return model

def train_model(X_train, y_train, X_test, y_test, X_val, y_val):

    # fetch cnn model
    model = nn_model()

    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)

    # Compile model
    model.compile(loss='binary_crossentropy',optimizer='adagrad',metrics=['accuracy'])

    weight_print = MyCustomCallback()
    # Fit model
    history = model.fit(X_train, y_train, epochs=200, batch_size=1000, validation_data=(X_val, y_val), verbose=1)#,callbacks = [weight_print])

    # plot ROC
    decisions_tf = model.predict(X_test)
    fpr_tf, tpr_tf, thresholds_tf = roc_curve(y_test, decisions_tf)
    auc_ = auc(fpr_tf,tpr_tf)
    fauc_ = "{:.2f}".format(auc_)
    figRoc = pyplot.figure(figsize=(15, 15))
    ax = pyplot.gca()    
    pyplot.text(0.5, 1.05, "CMS Simulation (Work In Progress)      (13 TeV)", fontweight="bold", horizontalalignment='center',verticalalignment='center', transform=ax.transAxes, fontsize=28)
    pyplot.plot(tpr_tf, 1-fpr_tf, linestyle='--',linewidth=8,color='blue', label='HH4tau vs ZZ - AUC:'+str(fauc_))
    #pyplot.plot([1, 0], [1, 0], linestyle='dotted', color='grey', label='Luck') # plot diagonal line
    pyplot.xlabel('Signal efficiency',fontsize=28)
    pyplot.ylabel('Background Rejection',fontsize=28)
    pyplot.legend(loc='best',fontsize=28)
    pyplot.grid()
    pyplot.savefig('ROC.png')

    # plot train-test comparisons
    compare_train_test_binary(model,X_train,y_train,X_test,y_test,'HH4tau vs ZZ')

    # evaluate model
    loss, acc = model.evaluate(X_test, y_test, verbose=1,batch_size=1000)
    print('Test loss: {:.4f}'.format(loss))
    print('Test accuracy: {:.4f}'.format(acc))

    # confusion matrix
    y_pred = model.predict(X_test)
    Y_test = y_test.reshape(len(y_test),1)
    Y_pred = y_pred
    Y_pred[y_pred<0.5] = 0
    Y_pred[y_pred>0.5] = 1 
    mat = confusion_matrix(Y_test, Y_pred)
    classes = [0,1]
    con_mat_norm = np.around(mat.astype('float') / mat.sum(axis=1)[:, np.newaxis], decimals=2)
    con_mat_df = pd.DataFrame(con_mat_norm, index=classes, columns=classes)

    # plot confusion matrix
    fig1 = pyplot.figure(figsize=(15, 15))
    ax = pyplot.gca()    
    pyplot.text(0.5, 1.05, "CMS Simulation (Work In Progress)      (13 TeV)", fontweight="bold", horizontalalignment='center',verticalalignment='center', transform=ax.transAxes, fontsize=28)
    sns.heatmap(con_mat_df, annot=True, cmap=pyplot.cm.Blues)
    #pyplot.tight_layout()
    
    pyplot.ylabel('True Class',fontsize=28)
    pyplot.xlabel('Predicted Class',fontsize=28)
    pyplot.savefig('confusion_matrix.png')

    # save trained model
    model.save('model.h5')

    return acc, history


def plot_model(history):
    # plot entropy loss
    pyplot.subplot(2, 1, 1)
    ax = pyplot.gca()    
    pyplot.text(0.5, 1.15, "CMS Simulation (Work In Progress)      (13 TeV)", fontweight="bold", horizontalalignment='center',verticalalignment='center', transform=ax.transAxes, fontsize=28)
    pyplot.title('Entropy Loss',fontsize=28)
    pyplot.plot(history[1].history['loss'], color='blue', label='train')
    pyplot.plot(history[1].history['val_loss'], color='red', label='test')

    # plot accuracy
    pyplot.subplot(2, 1, 2)
    pyplot.title('Accuracy',fontsize=28)
    pyplot.plot(history[1].history['accuracy'], color='blue', label='train')
    pyplot.plot(history[1].history['val_accuracy'], color='red', label='test')
    pyplot.xlabel('Epoch',fontsize=20)
    pyplot.savefig('loss_accuraccy.png')


def main():

    # 1 load train dataset
    X_train, y_train, X_test, y_test, X_val, y_val = prepare_data()

    # 2 train model
    history = train_model(X_train, y_train, X_test, y_test, X_val, y_val)

    # 3 plot model
    plot_model(history)

    print(('\033[1m'+'> Time Elapsed = {:.3f} sec'+'\033[0m').format((time.time()-start_time)))

if __name__ == "__main__":
    main()
