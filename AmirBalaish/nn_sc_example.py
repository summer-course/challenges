
# coding: utf-8

# In[2]:

import numpy as np
import pickle

samples = 10000
features = 10

fn = 'nn_XY.pickle'
with open(fn, 'wb') as f:
    pickle.dump(X, f)
    pickle.dump(Y, f)


# In[3]:

with open(fn, 'rb') as f:
    x = pickle.load(f)
    y = pickle.load(f)
    y = y[:, np.newaxis]
print 'loaded x with size {}'.format(x.shape)
print 'loaded y with size {}'.format(y.shape)



def sigm(t):
    return 1. / (1. + np.exp(-t))

def add_bias(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

def drop_bias(x):
    return x[:, :-1]

def simple_neural_network(x, layers_params):
    
    current_in = x.copy()
    
    for param in layers_params:
        out = np.dot(add_bias(current_in), param)
        out = sigm(out)
        current_in = out
        
    return out

def log_loss(y, y_hat):
    t = np.power(y_hat,y)*np.power(1 - y_hat,1 - y)
    t = np.log(t)
    return np.sum(t)


# In[5]:

_hidden_layer1 = 8
_hidden_layer2 = 4

layers = [np.random.normal(loc=0.0, scale=0.01, size=(features+1, _hidden_layer1)), 
          np.random.normal(loc=0.0, scale=0.01, size=(_hidden_layer1+1, _hidden_layer2)),
         np.random.normal(loc=0.0, scale=0.01, size=(_hidden_layer2+1, 1))]

y_hat = simple_neural_network(x, layers)
loss = log_loss(y, y_hat)

print loss
print y_hat.mean(), x.mean()


def calc_numeric_gradient(x,y,layer_params,i,j1,j2):
    yhat = simple_neural_network(x,layer_params)
    
    res1 = log_loss(y,yhat)
    eps = 0.0001 
    layer_params2 = [m.copy() for m in layer_params]
    layer_params2[i][j1,j2] += eps
    yhat2 = simple_neural_network(x,layer_params2)
    res2 = log_loss(y,yhat2)
    return (res2-res1)/eps
    

  
