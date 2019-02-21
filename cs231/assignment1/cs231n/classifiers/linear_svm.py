import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]#角标为y[i]的得分
    for j in xrange(num_classes): #计算样本在没有每一类的得分
      if j == y[i]:   #不计算本类别的loss  此时是本类别数据
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1 计算合页损失函数 
                          #其他类别在该类上的分数减去正确分数 分数差小于-1则是0 否则按照公式算
      if margin > 0:
        loss += margin
        dW[:,y[i]] += -X[i] # 我觉得只更新一次就行了 X[i]更新权重 减了
        dW[:,j] += X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train  # dW同样求均值

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)  # 直接对应元素相乘

  dW += 0.5 * reg * W # 什么用处?  后面消掉好看

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero



  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  num_train = X.shape[0]
  num_classes = W.shape[1]

  scores = X.dot(W) # 500*10
  correct_class_score = scores[np.arange(num_train),y] #500个
  correct_class_score = np.reshape(np.repeat(correct_class_score,num_classes),(num_train,num_classes))

  margin = scores - correct_class_score + 1.0
  margin[np.arange(num_train),y] = 0 #每一正确的类margin归0

  loss = (np.sum(margin[margin > 0]))/num_train
  loss += reg*np.sum(W*W)  # 这里我没有×系数0.5




  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margin[margin>0] = 1
  margin[margin<0] = 0

  row_sum = np.sum(margin,axis=1) 

  margin[np.arange(num_train),y] = -row_sum

  dW += np.dot(X.T,margin)
  dW /= num_train
  dW += reg * W  #什么用处
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
