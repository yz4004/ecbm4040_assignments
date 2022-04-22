import numpy as np
from random import shuffle

def sigmoid(x):
    """Sigmoid function implementation"""
    h = np.zeros_like(x)
    
    #############################################################################
    # TODO: Implement sigmoid function.                                         #         
    #############################################################################
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    
    h = 1./(1.+np.exp(-x))
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return h 

def logistic_regression_loss_naive(W, X, y, reg):
    """
      Logistic regression loss function, naive implementation (with loops)
      Use this linear classification method to find optimal decision boundary.

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where c can be either 0 or 1.
      - reg: (float) regularization strength. For regularization, we use L2 norm.

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)
    # print(dW.shape)   # (785,1)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.    #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the      #
    # regularization!                                        #
    #############################################################################
    #############################################################################
    #                     START OF YOUR CODE                  #
    #############################################################################
    
    N = X.shape[0]

    y_hat = sigmoid(np.dot(X,W))
    loss = - np.mean(y*np.log(y_hat) + (1-y)*np.log(1-y_hat)) 
    
    
    for i in range(N):

        y_hat = sigmoid(np.dot(X[i],W)) # (1,)

        k = -(y[i]-y_hat)*X[i]
        k = k.reshape(X.shape[1],1) # 785,1
        dW = dW + k
        
#         Li = - ( y[i]*np.log(y_hat) + (1-y[i])*np.log(1-y_hat)  )
#         loss = loss + Li

    # loss = loss/N + reg* np.sqrt(np.sum(W**2))   # + np.linalg.norm(W)
    loss = loss + reg* np.sqrt(np.sum(W**2)) 
    dW =  dW/N + 2*reg*W
    
    # print(loss,reg*np.sqrt(np.sum(W**2)))
    
    # print("./utils/classifiers/logistic_regression.logistic_regression_loss_naive() not implemented!") # delete me
    
    #############################################################################
    #                     END OF YOUR CODE                   #
    #############################################################################

    return loss, dW



def logistic_regression_loss_vectorized(W, X, y, reg):
    """
    Logistic regression loss function, vectorized version.
    Use this linear classification method to find optimal decision boundary.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Set the loss to a random number
    loss = 0
    # Initialize the gradient to zero
    dW = np.zeros_like(W)

    ############################################################################
    # TODO: Compute the logistic regression loss and its gradient using no     # 
    # explicit loops.                                                          #
    # Store the loss in loss and the gradient in dW. If you are not careful    #
    # here, it is easy to run into numeric instability. Don't forget the       #
    # regularization!                                                          #
    ############################################################################
    #############################################################################
    #                          START OF YOUR CODE                               #
    #############################################################################
    
    N = X.shape[0]
    
    f = np.dot(X,W) 
    
    sigmoid_f = sigmoid(f)
    
    # loss = - np.mean( (y*np.log(sigmoid_f) + (1-y)*np.log(1-sigmoid_f),axis=0) + reg* np.linalg.norm(W)  # reg*np.sqrt(np.sum(W**2))
    
    loss = np.mean( -(y*np.log(sigmoid_f) + (1-y)*np.log(1-sigmoid_f)) ) + reg*np.sqrt(np.sum(W**2))  
    y = y.reshape(sigmoid_f.shape)
    
    dW = np.mean(-(y - sigmoid_f)*X, axis= 0).reshape(W.shape)
    
    dW = dW + 2*reg*W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    

    return loss, dW
