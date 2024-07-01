
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:01:22 2023

@author: gbulb
"""




import seaborn as sns
import numpy as np
import pandas as pd
import tensorflow as tf
import glob
import ast
import torch
import torch.nn.functional as F
from math import sqrt, exp, log, pi
import re
import pickle
import random
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import os
import math
import csv
from datetime import datetime
from torchview import draw_graph
import gzip
from tensorflow.keras.utils import plot_model
from pandas.api.types import is_numeric_dtype
from torch.utils.data import DataLoader, random_split###
from numpy import loadtxt
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split,GridSearchCV, KFold, cross_val_score
from sklearn import metrics
from tensorflow.keras import initializers
from tensorflow.keras.initializers import GlorotNormal
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from tensorflow.keras import initializers
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score,accuracy_score
from sklearn.pipeline import Pipeline
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, Input, Add, Concatenate, concatenate
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D,Conv1D,Input, Flatten
from tensorflow.keras.layers import MaxPool2D,MaxPooling1D
from tensorflow.keras.layers import Flatten,Dropout,Dense
from sklearn.ensemble import RandomForestRegressor
import warnings
from yellowbrick.regressor import ResidualsPlot
warnings.filterwarnings("ignore")
from collections import defaultdict

# possible enhancements or additional features, this adds one more
enhance_with_total_count = True    # after the grid counts, add one dimension with total of all grid counts, 675+1
enhance_with_layer_0_count = False # after the grid counts, add one dimension with total count in layer 0, 675+2
enhance_with_one_hot_base = False   # after the grid counts and enhancements above, add four dimensions

# set path to build models for now
# results should go in that path as well

#path = "grid_count_675_7_1_1.5_3.5"   # 15x15x3 = 675 grid count data with width = 1.5
#path = "grid_count_363_5_1_1.0_3.5"   # 11x11x3 = 363 grid count data with width = 1.0
path = "grid_count_675_7_1_1.0_3.5"   # 15x15x3 = 675 grid count data with width = 1.0

# recover grid parameters from the path name
fields = path.split("_")
num_cells = int(fields[2])
hbin = int(fields[3])
vbin = int(fields[4])
width = float(fields[5])
height = float(fields[6])

print('Time %s. Starting' % datetime.now())

# read rfam family names
with open('C:/Users/gbulb/Dropbox/dropbox/PhD GBulbul/Rfam_to_standardized_name.txt','r') as f:
    lines = f.readlines()

rfam_to_standardized_name = defaultdict(str)
rfam_to_chain_count = defaultdict(int)
rfam_to_nt_count = defaultdict(int)

for line in lines:
    rfam, name = line.replace("\n","").split("\t")
    rfam_to_standardized_name[rfam] = name

files_list=[]

all_rfam = set()

for root, directories, files in os.walk(path):
    for name in files:

        # speed things up by using fewer files for testing
        #if random.random() < 0.0:
        #    continue

        if ".pickle" in name:
            files_list.append(os.path.join(root,name))

            rfam_family = os.path.split(name)[1].split("_")[0]
            all_rfam.add(rfam_family)

print('Identified %d data files from %d rfam families' % (len(files_list),len(all_rfam)))

#base_list = ["A"]              # use one base
#base_list = ["U"]              # use one base
base_list = ["A","C","G","U"]  # use all bases

keep_conservation_between = [0.0,0.9] # keep data points with low base conservation
keep_conservation_between = [0.9,1.1] # keep data points with high base conservation
keep_conservation_between = [0.0,1.1] # keep data points whose base conservation is between these numbers

# define groups of rfam families
SSU = set(["RF00177","RF01959","RF01960","RF02542","RF02545"])
LSU = set(["RF00002","RF02540","RF02541","RF02543","RF02546"])
assorted_small = set(["RF00004","RF00026","RF00167","RF01330","RF01344","RF01357","RF01051","RF01786","RF01734","RF01739","RF01704","RF00028","RF00009","RF00005","RF01750"])

#train_rfam = LSU | SSU  # roughly 90-10 split, but ribosomes are probably not a great guide to small RNAs
#train_rfam = all_rfam - LSU # train on everything except LSU chains
train_rfam = LSU | assorted_small # train on LSU and some small RNAs; test on SSU and other small RNAs

# separately define the rfam families to be testing data
test_rfam = all_rfam - train_rfam

# check for overlap
if len(train_rfam & test_rfam) > 0:
    print("Overlap between training and testing rfam families!!!  Stopping.")
#    print(stopnow)

# make a place to store training and testing data
train_context = []
train_conservation = []
train_one_hot_bases = []

test_context = []
test_conservation = []
test_one_hot_bases = []
test_unit_id=[]
# load each data file
for file in files_list:

    rfam_family = os.path.split(file)[1].split("_")[0]

    # skip any data files that are not in train or test
    if not rfam_family in train_rfam | test_rfam:
        continue

    try:
        with open(file, 'rb') as f_obj:
            chain_data = pickle.load(f_obj)
        print("Read %5d data points from %s" % (len(chain_data),file))
    except:
        print('unable to load %s:' % file)
        continue

    rfam_to_chain_count[rfam_family] += 1

    # loop over each nucleotide in the file
    for unit_id,context_count,base_conservation,max_letter in chain_data:
        # leave out some data points, as desired

        # if a current_base is defined and this one does not match, skip this data pair
        base = unit_id.split("|")[3]
        if not base in base_list:
            continue

        # skip base conservation values outside of the range
        if base_conservation < keep_conservation_between[0] or base_conservation > keep_conservation_between[1]:
            continue

        # count nucleotides used from each rfam family
        rfam_to_nt_count[rfam_family] += 1

        # enhance the data if desired
        if enhance_with_total_count:
            sum_context_count = sum(context_count)
            context_count.append(sum_context_count)   # add one more predictor

        if enhance_with_layer_0_count:
            layer_0_start = (2*hbin+1)*(2*hbin+1)*vbin
            layer_0_end   = (2*hbin+1)*(2*hbin+1)*(vbin+1)
            sum_layer_0 = sum(context_count[layer_0_start:layer_0_end])
            context_count.append(sum_layer_0)   # add one more predictor

        if base == 'A':
            one_hot = [1,0,0,0]
        elif base == 'C':
            one_hot = [0,1,0,0]
        elif base == 'G':
            one_hot = [0,0,1,0]
        elif base == 'U':
            one_hot = [0,0,0,1]
        else:
            one_hot = [0,0,0,0]

        if enhance_with_one_hot_base:
            context_count += one_hot

        # split between training and testing data using rfam family
        if rfam_family in train_rfam:
            train_context.append(context_count)
            train_conservation.append(base_conservation)
            train_one_hot_bases.append(one_hot)

        if rfam_family in test_rfam:
            test_context.append(context_count)
            test_conservation.append(base_conservation)
            test_one_hot_bases.append(one_hot)

# tell how many data points are in the training set and the testing set
for rfam_family in sorted(train_rfam, key = lambda x: rfam_to_standardized_name[x].upper()):
    if rfam_family in train_rfam:
        print("%s provided %6d data points from %3d chains for training; %s" % (rfam_family,rfam_to_nt_count[rfam_family],rfam_to_chain_count[rfam_family],rfam_to_standardized_name[rfam_family]))

for rfam_family in sorted(test_rfam, key = lambda x: rfam_to_standardized_name[x].upper()):
    if rfam_family in test_rfam:
        print("%s provided %6d data points from %3d chains for testing; %s" % (rfam_family,rfam_to_nt_count[rfam_family],rfam_to_chain_count[rfam_family],rfam_to_standardized_name[rfam_family]))


print('Dataset folder: %s' % path)
#### record which Rfam families went where
if len(base_list) == 1:
    print('Using base %s' % base_list[0])
else:
    print('Using bases %s' % base_list)

print('Using data points with base conservation between %0.2f and %0.2f' % (keep_conservation_between[0],keep_conservation_between[1]))

if enhance_with_total_count:
    print('Added one dimension for the total count')

if enhance_with_layer_0_count:
    print('Added one dimension for the layer 0 count')

if enhance_with_one_hot_base:
    print('Added four dimensions for one-hot encoding of the base')

#print('There are %d predictors' % len(train_context[0]))

print('Training set has %5d data points' % len(train_context))
print('Testing  set has %5d data points' % len(test_context))

# use the data to train and test some models below
#################X & Y preprocessing X:context, y:conservation score #################
class preprocessing:
      def pre_X_contexts(train_context,test_context):
          X_train_context, X_test_context= 1.0*np.array(train_context), np.array(test_context)    
          scaler = StandardScaler()
          X_train = scaler.fit_transform(X_train_context)
          X_test = scaler.transform(X_test_context)# seems to help, replace integers with floats
          return X_train_context, X_test_context
      def pre_Y_conservation(train_conservation,test_conservation):
          y_train,y_test = np.array(train_conservation),np.array(test_conservation)
          return y_train,y_test
      def pre_one_hot(train_one_hot_bases,test_one_hot_bases):
          one_hot_train,one_hot_test=np.array(train_one_hot_bases),1.0*np.array(test_one_hot_bases)
          return one_hot_train,one_hot_test
vec_train,vec_test=preprocessing.pre_one_hot(train_one_hot_bases,test_one_hot_bases)
print('vec_train:', vec_train)
X_train,X_test=preprocessing.pre_X_contexts(train_context,test_context)
print('X_train:', X_train.shape[1])
y_train,y_test=preprocessing.pre_Y_conservation(train_conservation,test_conservation)
############################Reshaping input#######################################
class reshaping:
    def rehape_X_train(X_train):
          
          sample_size,input_steps,input_dimension = X_train.shape[0], X_train.shape[1] ,1 
          #input_shape =(number of samples in train set,number of features in train set, each feature is represented by 1 number)
          #input_shape = (sample_size,time_steps,input_dimension)
          train_data_reshaped = X_train.reshape(sample_size,input_steps,input_dimension)
          return train_data_reshaped

    def arranging_input(embedding_size=1,no_of_unique_cat  =4): #no_of_unique_cat  = len(np.unique(one_hot))
        train_data_reshaped=reshaping.rehape_X_train(X_train)
        n_timesteps = train_data_reshaped.shape[1] #13
        n_features  = train_data_reshaped.shape[2] #1 
        num_input = Input(shape=(n_timesteps,n_features))
        inp_cat_data = Input(shape=(no_of_unique_cat,))
        inp_num_data = Input(shape=(train_data_reshaped.shape[1],))
        inputs=[inp_cat_data, inp_num_data]
        emb = Embedding(input_dim=no_of_unique_cat, output_dim=embedding_size)(inp_cat_data)  
        return emb,inp_cat_data,num_input
emb,inp_cat_data,num_input=reshaping.arranging_input(embedding_size=1,no_of_unique_cat  =4)
flatten = Flatten()(emb)
###----3rd
'''
y=Conv1D(filters=128, kernel_size=7, kernel_initializer=initializer, activation=activation1, name="Conv1D_1")(num_input)

y=Conv1D(filters=64, kernel_size=3, kernel_initializer=initializer, activation=activation1, name="Conv1D_2")(y)
  
y=Conv1D(filters=32, kernel_size=2, kernel_initializer=initializer, activation=activation1, name="Conv1D_3")(y)
y=MaxPooling1D(pool_size=2, name="MaxPooling1D")(y)
y=Conv1D(filters=16, kernel_size=3, kernel_initializer=initializer, activation=activation1, name="Conv1D_4")(y)
 
y=Conv1D(filters=8, kernel_size=2, kernel_initializer=initializer, activation=activation1, name="Conv1D_5")(y)
'''
####----1st
'''
y=Conv1D(filters=64, kernel_size=7, kernel_initializer=initializer, activation=activation1, name="Conv1D_1")(num_input)

y=Conv1D(filters=32, kernel_size=3, kernel_initializer=initializer, activation=activation1, name="Conv1D_2")(y)
  
y=Conv1D(filters=16, kernel_size=2, kernel_initializer=initializer, activation=activation1, name="Conv1D_3")(y)
y=MaxPooling1D(pool_size=2, name="MaxPooling1D")(y)
y=Conv1D(filters=8, kernel_size=3, kernel_initializer=initializer, activation=activation1, name="Conv1D_4")(y)
 
y=Conv1D(filters=4, kernel_size=2, kernel_initializer=initializer, activation=activation1, name="Conv1D_5")(y)
'''

#################Modeling##################
class Conv1D_model(Sequential):
    def __init__(self, model_params):
        super().__init__()
        self.add(Input(model_params['num_input']))
        self.add(Conv1D(filters=64, kernel_size=7, kernel_initializer=model_params['initializer'], activation=model_params['activation1'], name="Conv1D_1")),
        self.add(Conv1D(filters=64, kernel_size=2, kernel_initializer=model_params['initializer'], activation=model_params['activation1'], name="Conv1D_3"))
        self.add(MaxPooling1D(pool_size=2, name="MaxPooling1D-3")),
        self.add(Conv1D(filters=8, kernel_size=3, kernel_initializer=model_params['initializer'], activation=model_params['activation1'], name="Conv1D_4")),
        self.add(Conv1D(filters=4, kernel_size=2, kernel_initializer=model_params['initializer'], activation=model_params['activation1'], name="Conv1D_5")),
        self.add(Flatten())
        self.y=Flatten()
        self.x=Flatten((model_params['emb']))
        z= Concatenate()([self.x, self.y])
        out = Dense(1)(z)
        model = Model(inputs=model_params['inputs'], outputs=out)
        return model
        

model_params={
    'inputs':[inp_cat_data, num_input],
    'num_input':num_input,
    'emb':emb,
'activation1':'relu',
'batch_size1':64,
'initializer' : GlorotNormal(seed=0)
}
model = Conv1D_model(model_params)
class get_predictions_after_fitting:
      def get_predictions(vec_train,X_train,y_train,vec_test,X_test):
          history=model.fit([vec_train,X_train], y_train, epochs=10, batch_size=32)
          ypred = model.predict([vec_test,X_test])
          return ypred
#plot_model(model, to_file='model_1dcnn_680_concat_3_conv_plus_cat_3_new_3_5.10_hidden_4.png',show_shapes=True, show_layer_names=True)
################Model evaluation-Metrics#################
y_pred=get_predictions_after_fitting.get_predictions(vec_train,X_train,y_train,vec_test,X_test)
###
y_pred_list = [max(0.25,min(1.0,yp[0])) for yp in y_pred] 
print('Mean response',sum(y_train)/len(y_train))
for i in range(0,10):
    print('actual: %8.4f %8.4f predicted' % (y_test[i],y_pred_list[i]))

print(min(y_pred_list),max(y_pred_list),len(y_pred_list),len(y_test))

predictions=y_pred_list

class get_metrics:
      def MAE_(y_test,predictions):
          return mean_absolute_error(y_test, predictions)
      
      def MSE_(y_test,predictions):
          return mean_squared_error(y_test, predictions)
      
      def r2_score_(y_test,predictions):
          return r2_score(y_test, predictions)
print("MAE: %.4f" % get_metrics.MAE_(y_test,predictions)) 
print("MSE: %.4f" % get_metrics.MSE_(y_test,predictions))
print("r2: %.4f" % get_metrics.r2_score_(y_test,predictions))
################Model evaluation-Plot diagnostics#################
ypred=predictions
list_of_i=[]
list_of_j=[]
for i in ypred:
    i=float(i)
    list_of_i.append(i)

#print('list_of_i:',list_of_i)
#print('ytest:',ytest)
ypred=np.asarray(list_of_i)
#ytest=np.asarray(list_of_j)
#ytest=y_test.to_numpy() 
for j in y_test:
    j=float(j)
    list_of_j.append(j)
y_test=np.asarray(list_of_j)
predictions=ypred
fig, ax = plt.subplots()
ax=sns.scatterplot(y_test, predictions, hue=(abs(y_test-predictions)>=0.3),marker='.')
plt.title("1D-CNN on 680 Grid Count dataset", fontsize = 15)
 # title of scatter plot
plt.xlabel("True Base Conservation Scores", fontsize = 15) # x-axis label
plt.xlim(0.20, 1.1)
plt.ylim(0.20,1.1)
plt.ylabel("Predicted Base Conservation Scores", fontsize = 15) # y-axis label
custom = [Line2D([], [], marker='.', color='g', linestyle='None'),
          Line2D([], [], marker='.', color='b', linestyle='None')]

plt.legend(custom, ['above 0.3', 'below 0.3'], loc='lower right')

#plt.axis('equal')
plt.show()
################Model evaluation-Detecting unusual observations#################
ypred=predictions
def discrepancy_ypred_ytest(ypred,ytest):
    return ypred-ytest
print("All index value of >0.3 is: ", np.where(abs(discrepancy_ypred_ytest(ypred,y_test))>0.3)[0])
print('yindex:', abs(discrepancy_ypred_ytest(ypred,y_test))>0.3)
print('y_test_index:', y_test[abs(discrepancy_ypred_ytest(ypred,y_test))>0.3])
print('ypred_index:', ypred[abs(discrepancy_ypred_ytest(ypred,y_test))>0.3])
