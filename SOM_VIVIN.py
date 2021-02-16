# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:38:16 2020

@author: vivin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
os.chdir(r"D:\Work\Udemy\DeepLEarningA-Z\Resource files\Self_Organizing_Maps")

dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
X = sc.fit_transform(X)

#Training the SOM
from minisom import MiniSom
som = MiniSom(x=10,y=10, input_len=15,sigma=1.0,learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X,num_iteration=100)

#Visualising the data
from pylab import plot,bone,pcolor,colorbar,show
bone()
pcolor(som.distance_map().T)
colorbar()
#outliers
#red circle = customer did not get approval
#green square = cutomer got approval
markers=['o','s']
colors=['r','g']

for i,x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5, w[1]+0.5,markers[y[i]],markeredgecolor=colors[y[i]],
         markerfacecolor='None',markersize=10,markeredgewidth=2)
show()

#finding the fraudsters
mappings = som.win_map(X)
frauds = np.concatenate((mappings[(1,7)],mappings[(8,5)]),axis=0)
frauds = sc.inverse_transform(frauds) #Final list of fraudsters.
