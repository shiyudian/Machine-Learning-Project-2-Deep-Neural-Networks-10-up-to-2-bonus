from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import pickle

import os
from scipy.io import loadmat
# -----------------------------------
from copy import copy, deepcopy
import shutil
import csv
import tempfile
#import sys
import subprocess
#import datetime
import random
import scipy.stats

#Load the MNIST digit data
M = loadmat("mnist_all.mat")


def softmax(y):
    '''Return the output of the softmax function for the matrix of output y. y
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(y)/tile(sum(exp(y),0), (len(y),1))   #P_i 
    
def tanh_layer(y, W, b):    
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1) #P_i
    return L0, L1, output


def cross_entropy(y, y_):
    return -sum(y_*log(y))   #C cost

    

#Load sample weights for the multilayer neural network
snapshot = pickle.load(open("snapshot50.pkl"))
W0 = snapshot["W0"]
b0 = snapshot["b0"].reshape((300,1))
W1 = snapshot["W1"]
b1 = snapshot["b1"].reshape((10,1))

#Load one example from the training set, and run it through the
#neural network
x = M["train5"][148:149].T    
L0, L1, output = forward(x, W0, b0, W1, b1)
#get the index at which the output is the largest
y = argmax(output)

################################################################################
#Code for displaying a feature from the weight matrix mW
#fig = figure(1)
#ax = fig.gca()    
#heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)    
#fig.colorbar(heatmap, shrink = 0.5, aspect=5)
#show()
################################################################################
    
def part1():
    for seed in range (0, 10):
        fig = figure()
        for i in range(0, 10):
            index = np.random.randint(0, 5500)
            ax = fig.add_subplot(1+(i//5),5, 1+(i%5))
            imshow(M["train"+str(seed)][index].reshape((28,28)), cmap=cm.gray)
            plt.axis('off')
        plt.savefig("digits_pics/10_img_of_"+str(seed)+".jpg", bbox_inches='tight')
    show()
 
            
def calculateOutput(weights, x):
    out = np.dot(x, weights[:784, :]) + weights[-1, :]
    return softmax(out)


def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network
    
    y = y_i     where i in range (0,9), y is actual hot-code
    y_ = P_i    where j in range (1,784)
    
    L1 = o_i
    L0 = hidden layer w/ 300 units
    x = inputs in range [1,784]
    '''
    dCdL1 =  y - y_     #P_i - y_i = dCdo_i
    dCdW1 =  dot(L0, dCdL1.T ) #dL1dL0
    # b0,b1 biases are constants  
    
    print dCdL1.shape
    print dCdW1.shape

#--------------PART 2 - COST FUNC ---------------------------

def forwardPropagation(inputs, W, b):  
    """inputs: matrix of shape (784, N)
    Output: Prediction.shape (10, N)"""     
    o = np.dot(W.T, inputs) + b
    predic = softmax(o)  # Output prediction. 
    return predic 

def part2_cost_function(target, x, w, b):  
    predic = forwardPropagation(x,w,b)
    return -np.sum(target*np.log(predic))

##
def part3_gradient(train_inputs, predic, train_targets, W, b):
    '''
    Inputs' shape: (784, 50) (10, 50) (10, 50) (784, 10) (10, 1)
    Output shape: (784, 10) (10, 1)
    '''
    # Compute derivation 
    dCbydo = predic - train_targets
    
    # Backpropagation
    dCbydh_output = np.dot(W, dCbydo)
    dCbydh_input = dCbydh_output * train_inputs * (1 - train_inputs)

    # Gradients for weights and biases.
    dCbydW = np.dot(train_inputs, dCbydo.T)
    dCbydb = np.sum(dCbydo, axis=1).reshape(-1, 1)    
   
    return dCbydW, dCbydb


def make_set(data_set, train_size, test_size):
    '''
    helper function used to generate the training input, training target, test
    input, test target used the given size.
    
    returns:
    train_inputs: train_size*10 x 784 matrix
    train_targets: train_size*10 x 10 matrix 
    test_inputs: test_size*10 x 784 matrix
    test_targets: test_size*10 x 10 matrix
    '''
    #train_size 50/10=5.

    test_targets=np.zeros((test_size*10, 10))
    train_targets=np.zeros((train_size*10, 10))
    
    for i in range(10):
        key_train='train'+str(i)
        key_test='test'+str(i)

        train_set=[n for n in range(data_set[key_train].shape[0])]
        test_set=[m for m in range(data_set[key_test].shape[0])]
        
        pick_train=random.sample(train_set,train_size)
        pick_test=random.sample(test_set,test_size)
        
        new_train_inputs=data_set[key_train][pick_train,]/255.0
        new_test_inputs=data_set[key_test][pick_test,]/255.0
        
        if i==0:
            train_inputs=new_train_inputs
            test_inputs=new_test_inputs
        else:
            train_inputs=np.concatenate((train_inputs,new_train_inputs), axis=0)
            test_inputs=np.concatenate((test_inputs,new_test_inputs), axis=0)
                        
        train_targets[i*train_size:(i+1)*train_size,i]=1
        test_targets[i*test_size:(i+1)*test_size,i]=1
    return train_inputs, train_targets, test_inputs, test_targets   

##
#        num_inputs=N
#        num_hiddens=300        
#        num_outputs=10 
def InitNN(num_inputs, num_hiddens, num_outputs):
    '''Initializes NN parameters.'''
    W1 = 0.01 * np.random.randn(num_inputs, num_hiddens)
    W2 = 0.01 * np.random.randn(num_hiddens, num_outputs)
    b1 = np.zeros((num_hiddens, 1))
    b2 = np.zeros((num_outputs, 1))
    return W1, W2, b1, b2

def TrainNN_softmax(M, EPS, r, num_iter):    
    train_inputs, train_targets, test_inputs, test_targets = make_set(M, 100, 50)
    
    train_inputs = train_inputs.T
    test_inputs = test_inputs.T
    train_targets = train_targets.T
    test_targets = test_targets.T
    
    W, W1, b, b1 = InitNN(train_inputs.shape[0], train_targets.shape[0], 0)
    
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)
    num_to_train = train_inputs.shape[1]
    
    train_rates_set = []
    train_costs_set = []
    test_rates_set = []
    test_costs_set = []

#----
    for iter in range(num_iter):
        predic = forwardPropagation(train_inputs, W, b)    
        dCbydW, dCbydb = part3_gradient(train_inputs,predic,train_targets,W,b)    

        dW = r* dW - (EPS / num_to_train) * dCbydW
        db = r* db - (EPS / num_to_train) * dCbydb
    
        W = W + dW
        b = b + db
        
        train_mp= (np.argmax(train_targets.T,axis=1)==np.argmax(predic.T,axis=1)).mean() 
        train_cost = part2_cost_function(train_targets, train_inputs, W, b)
        train_rates_set.append(train_mp)
        train_costs_set.append(train_cost)

        test_predic = forwardPropagation(test_inputs, W, b) 
        test_mp= (np.argmax(test_targets.T,axis=1)==np.argmax(test_predic.T,axis=1)).mean() 
        test_cost = part2_cost_function(test_targets, test_inputs, W, b)
        test_rates_set.append(test_mp)
        test_costs_set.append(test_cost)     
        
    return W, b, train_inputs, predic, train_targets, train_rates_set, train_costs_set, test_inputs, test_predic, test_targets, test_rates_set, test_costs_set



def part3b_check_gradient_finite_diff(predic_f,gradient_f,cost_f,target, x,w,b,delta,i,j):
    dw,db=np.zeros(w.shape),np.zeros(b.shape)
    dw[j,i], db[j,0] = delta, delta
    
    predic = predic_f(x.T, w.T,b)
    
    actual_wij = ((cost_f(target.T, x.T, w.T+dw.T, b)-cost_f(target.T, x.T, w.T-dw.T,b))/2*float(delta))*1e+12
    actual_bj = ((cost_f(target.T, x.T, w.T, b+db)-cost_f(target.T, x.T, w.T,b-db))/2*float(delta))*1e+12
    
    pick_w, pick_b=gradient_f(x.T, predic, target.T, w.T, b)
    pick_wij, pick_bj=pick_w[i,j], pick_b[j,0]
    
    print "\n i, j =", i, j
    print "\n actual wij", actual_wij, "\n computed wij", pick_wij, "\n actual bj", actual_bj, "\n computed bj", pick_bj
    print '\n diff wrt wij:', actual_wij-pick_wij,'\n diff wrt bj:',actual_bj-pick_bj



def p3b(M, i_list=[210, 320], j_list=[2,6]):
    train_inputs, train_targets, test_inputs, test_targets = make_set(M, 5, 5)

    w = 0.01 * np.random.randn(10,train_inputs.shape[1])
    b = 0.01 * np.random.randn(10,1)
    
    
    TrainNN_softmax(M, 0.01,1, 100)
    for i in i_list:
        for j in j_list:
            part3b_check_gradient_finite_diff(forwardPropagation,part3_gradient,part2_cost_function,train_targets, train_inputs, w,b,0.000001,i,j)  

#****************************************************
print("\n--------------\n    PART 3b\n--------------")
p3b(M)

##              PART 4 PLOT LEARNING CURVE

            
def part4(M,num_iter):
    xs=[i for i in range(num_iter)]
    W, b, train_inputs, predic, train_targets, train_rates_set, train_costs_set, test_inputs, test_predic, test_targets, test_rates_set, test_costs_set = TrainNN_softmax(M, 0.01,1,num_iter)
#    print W.shape[1]
#    print W
         
    for i in range(len(W[1])):
        img=W[:,i].reshape((28,28))
        plt.imshow(img,cmap=cm.coolwarm)
        imsave('weight' +str(i),img) 
           
    plt.clf()
    plt.plot(xs, test_rates_set, 'g', label="Test accuracy")
    plt.plot(xs, train_rates_set, 'r', label='Train accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel("Number of Iterations")
    plt.title("Learning curve")    
    plt.legend(loc = 'best')
    plt.savefig('part4_learning_curve.jpg')
    
#****************************************************    
print("\n--------------\n    PART 4\n--------------")
# iteration 200
part4(M,200)
print "check images saved in folder"

##             PART 5  

def plot_line(theta, x_min, x_max, color, label):
    x_grid_raw = arange(x_min, x_max, 0.01)
    x_grid = vstack((    ones_like(x_grid_raw),
                         x_grid_raw,
                    ))
    y_grid = dot(theta, x_grid)
    plot(x_grid[1,:], y_grid, color, label=label)

def gradf(f,t,error=1e-11):
  t = asarray(t)
  return asarray((f(t+error)-f(t-error))/(2*error))

def grad_descent2(f,init_t, alpha=1e-6):
  init_t = asarray(init_t)
  if alpha<1e-4:
    alpha = 1e-4

  EPS = 1e-8
  prev_t = init_t.copy()

  prev_t = (init_t-10*EPS).copy()
  t = init_t.copy()

  max_iter = 5000
  iter_n = 0

  residue0 = norm(t - prev_t)
  residue1 = norm(t - prev_t)

  factor0 = 1.22
  b_fac = 1.3
  factor1 = 0.5

  for i in range(max_iter):
  #while  residue1 > EPS and iter_n < max_iter:
    residue0 = residue1

    residue0 = residue1
    prev_t = t.copy()
    #t -= gradf(f,t)*alpha
    t -= gradf(f,t)*alpha
    iter_n += 1

    residue1 = norm(t - prev_t)
    #if f(t)-f(prev_t) < ala_r*dot(t-pre_t,t-pre_t)/alpha:
    if residue1 < residue0*factor0:
      #print "Boost"
      alpha = alpha*b_fac
      if alpha> 0.5:
        alpha = 0.5

    if f(t) > f(prev_t):
      #print "Slow"
      if norm(t)*alpha > EPS*100:
        alpha = alpha*factor1
        t = prev_t.copy()

    if residue1 < EPS:
      break

  if iter_n >= max_iter-1:
    print [alpha,t,residue1]
    print "Overloaded"
  return [alpha,t]

#print grad_descent2(cos,3.0,3e-5)

def gen_lin_data_1d(theta, N, sigma):

    # Actual data
    x_raw = 100*(np.random.random((N))-.5)
    
    x = vstack((    ones_like(x_raw),
                    x_raw,
                    ))
                
    y = dot(theta, x) + scipy.stats.norm.rvs(scale= sigma,size=N)
    #saveJ(x,"tem_x")
    #saveJ(x,"tem_y")

    plot(x[1,:], y, "ro", label = "Training set")

    # Actual generating process

    plot_line(theta, -70, 70, "b", "Actual generating process")

    # Least squares solution
    theta_hat = dot(linalg.inv(dot(x, x.T)), dot(x, y.T))
    plot_line(theta_hat, -70, 70, "g", "Maximum Likelihood Solution")

    def C_fun(a_0):
      tem_l = maximum(0.0,1/(1+abs(dot(asarray(a_0), x))))
      tem_y = maximum(0.0,1/(1+abs(y)))
      return -sum(dot(tem_y,log(tem_l))+dot((1-tem_y),log(1-tem_l)))


    a,theta = grad_descent2(C_fun,(-3.1,1.1))
    plot_line(theta, -70, 70, "r", "Multinomial Logistic Regression")


    legend(loc = "best")
    xlim([-70, 70])
    ylim([-100, 100])

    
    
#****************************************************    
print("\n--------------\n    PART 5\n--------------")

print "check images saved in folder"

theta = array([-3, 1.5])
N = 10
sigma1 = 10.2
sigma2 = 120.2

plt.clf()
gen_lin_data_1d(theta, N, sigma1)
plt.savefig("part5_sigma_small")

plt.clf()
gen_lin_data_1d(theta, N, sigma2)
plt.savefig("part5_sigma_big")
plt.clf()





   