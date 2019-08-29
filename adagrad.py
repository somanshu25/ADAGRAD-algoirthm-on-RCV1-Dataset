#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.datasets import fetch_rcv1 
import numpy as np
import random
from numpy import linalg
from scipy.sparse import diags,identity

rcv1 = fetch_rcv1()

rcvLabel = rcv1.target.getcol(33)
rcvLabel = rcvLabel.astype(np.float)
rcvLabel[rcvLabel == 0] = -1

xData_train = rcv1.data[:100000,:]
yLabel_train = rcvLabel[:100000]

xData_test = rcv1.data[100001:, :]
yLabel_test = rcvLabel[100001:]





# In[3]:


regPar = 0.0000010
w = np.zeros((rcv1.data.shape[1], 1), dtype = float)
B = 1000
i = 100


# In[21]:


# print(w.shape)
G = identity(rcv1.data.shape[1]).tocsr()
for t in range(1, 100):
    
    sample_indices = random.sample(range(0, xData_train.shape[0]-1), B)
    x_sample = xData_train[sample_indices]
    y_sample = yLabel_train[sample_indices]
    predictXY = x_sample.dot(w)
    false_indices_map = y_sample.multiply(predictXY) < 1
    false_indices  = np.where(false_indices_map.todense())[0]
    x_sample_falseclassified =  x_sample[false_indices]
    y_sample_falseclassified =  y_sample[false_indices]
    yx = np.sum(y_sample_falseclassified.multiply(x_sample_falseclassified), axis=0)
    #print(yx.shape)
    
    
    gradient = np.dot(regPar, w) - yx.transpose()/B 
    nue = 1.0/(regPar*i)

    print("Iteration",t)
    predictionsTrain = np.where((yLabel_train.multiply(xData_train.dot(w)) < 1).todense())[0]
    print("Training Error", predictionsTrain.shape)
    #print(gradient.shape)
    #test_predictions = np.where((y_test.multiply(x_test.dot(w.transpose())) < 1).todense())[0]
    #print("Test Error", test_predictions.shape)
    
    diag_G = G.diagonal().reshape(1,rcv1.data.shape[1])
    G_inv = diags(np.reciprocal(diag_G),[0]).tocsr()
    w1 = w - nue*(G_inv.dot(gradient))
    w = min(1, 1/((linalg.norm(G*w1-w1))*np.sqrt(regPar))) * w1
    #print("hello")
    diag_G_ele = np.square(diag_G)
    gradient_ele = np.square(gradient)
    sum_grad = np.array(np.sqrt(diag_G_ele.transpose() + gradient_ele))
    G = diags(sum_grad.transpose(), [0]).tocsr()


# In[17]:





# In[15]:





# In[18]:





# In[ ]:




