# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 21:59:08 2026

@author: reini
"""

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as f
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#--------- syntehtic data --------------
# x= np.linspace(0,1000, 1000)

# vectors = np.array([
#     np.sin(x),
#     np.sin(2*x),
#     np.sin(3*x),
#     np.sin(4*x)
# ])
#---------------------------------------

#----------------Real Data

def load_omni(file_name):

    df = pd.read_csv(file_name, sep='\t')    # load file into dataframe

    #dictionary for missing data values for selected columns
    error_cd ={
        "B":999.9,
        "Bx":999.9,
        "By":999.9,
        "Bz":999.9,
        "T":9999999,
        "N":999.9,
        "V":9999,
        "P":99.99,
        "Kp":99,
        "AE":9999,
        "Dst":99999,
        "SSN":999,
        "f10_7":999.9
        }
    #replace missing value with values from dictionary
    for column in df.columns[1:len(df.columns)]:
        df[column] = df[column].replace(error_cd[column], np.nan)
    # convert the date to Python datetime format 
    df['Date'] = pd.to_datetime(df['Date'])
    return df


data = load_omni('omni_data_fmt.csv')[0:5000].dropna(axis=0)

x_dst= np.array(data[['Dst']]).T #(datapoints, 1)
x_b= np.array(data[['Bx','By','Bz']]).T #(data points, num of variables)

sc_dst= MinMaxScaler()
sc_b= MinMaxScaler()
normalized_data_dst=sc_dst.fit_transform(x_dst)
normalized_data_b=sc_b.fit_transform(x_b)




####################################################
X_dst=torch.tensor(normalized_data_dst,dtype=torch.float32) #tokens, features
X_b=torch.tensor(normalized_data_b,dtype=torch.float32) #tokens, features

d_model = X_dst.shape[-1] #dimension of features


d_k= 500 #will be layer diminsion, must be smaller thannum of data points or "features"


w_q=nn.Linear(d_model,d_k)
w_k=nn.Linear(d_model,d_k)
w_v= nn.Linear(d_model,d_k)



q= w_q(X_dst)
k=w_k(X_b)
V= w_v(X_b)

s=(q @ k.T)/(d_k**0.5) #how tokens realte to each other



A= f.softmax(s, dim=-1)

Y= A @ V
# print(A)
# print(Y)

#plots
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure()
plt.title("A: Correlation of Tokens")
sns.heatmap(A.data, annot=True)
plt.show()

plt.figure()

# for i,y in enumerate(Y.data): 
#     plt.subplot(Y.data.shape[0],1, i+1)
#     plt.plot(y[0:100]*1000)
# plt.show()
# plt.figure()
# for i,y in enumerate(Y.data): 
    
#     plt.plot(y[0:100], alpha=0.5)
# plt.show()





