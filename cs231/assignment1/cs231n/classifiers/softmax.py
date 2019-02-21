import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  dW = np.zeros_like(W)    # D by C
  dW_each = np.zeros_like(W)
  num_train, dim = X.shape
  num_class = W.shape[1]
  f = X.dot(W)    # N by C
  # Considering the Numeric Stability
  f_max = np.reshape(np.max(f, axis=1), (num_train, 1))   # N by 1
  prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True) # N by C
  y_trueClass = np.zeros_like(prob)
  y_trueClass[np.arange(num_train), y] = 1.0
  for i in xrange(num_train):
      for j in xrange(num_class):    
          loss += -(y_trueClass[i, j] * np.log(prob[i, j]))    
          dW_each[:, j] = -(y_trueClass[i, j] - prob[i, j]) * X[i, :]
      dW += dW_each
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W


  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_class = W.shape[0]
  num_train = X.shape[0]
  scores = X.dot(W)
  maxLogc = np.reshape(np.max(scores,axis=1),(num_train,1)) #计算maxLogc
  prob = np.exp(scores - maxLogc)/np.sum(np.exp(scores - maxLogc),axis=1,keepdims=True)#计算出每一类概率值
  correct_class = np.zeros_like(prob)
  correct_class[np.arange(num_train),y] = 1 #花式索引 概率为1。y的值为label 作为索引
  loss = -np.sum(correct_class * np.log(prob)) / num_train + 0.5 * reg * np.sum(W * W)
  dW = -np.dot(X.T, correct_class - prob) / num_train + reg * W
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

