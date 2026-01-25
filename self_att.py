import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as f


x= np.linspace(0,1000, 1000)

vectors = np.array([
    np.sin(x),
    np.sin(2*x),
    np.sin(3*x),
    np.sin(4*x)
])
 #tokens, features

X=torch.tensor(vectors,dtype=torch.float32)

d_model = vectors.shape[-1] #dimension of features


d_k= d_model #will be layer diminsion


w_q=nn.Linear(1000,d_k)
w_k=nn.Linear(1000,d_k)
w_v= nn.Linear(1000,d_k)



q= X @ w_q.weight.T
k=X @ w_k.weight.T
V= X@ w_v.weight.T
s=q @ k.T #how tokens realte to each other

s/=torch.sqrt(torch.tensor(d_k,dtype=s.dtype) )

A= f.softmax(s, dim=-1)

Y= A @ V
print(A)
print(Y)

