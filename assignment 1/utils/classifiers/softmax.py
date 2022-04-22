import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
      Softmax loss function, naive implementation (with loops)
      This adjusts the weights to minimize loss.

      Inputs have dimension D, there are C classes, and we operate on minibatches
      of N examples.

      Inputs:
      - W: a numpy array of shape (D, C) containing weights.
      - X: a numpy array of shape (N, D) containing a minibatch of data.
      - y: a numpy array of shape (N,) containing training labels; y[i] = c means
        that X[i] has label c, where 0 <= c < C.
      - reg: (float) regularization strength. For regularization, we use L2 norm.

      Returns a tuple of:
      - loss: (float) the mean value of loss functions over N examples in minibatch.
      - gradient: gradient wrt W, an array of same shape as W
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
    #############################################################################
    #                     START OF YOUR CODE                                    #
    #############################################################################
    
    # y=y.reshape(len(y),1)

    N = X.shape[0]  # 100
    C= W.shape[1]   # 20
    
    
    # Y = np.zeros((N,C))
    
    for i in range(N):
        
        O = X[i].dot(W)
        O = O - max(O)
        softmax = np.exp(O)/sum(np.exp(O))
        loss = loss - np.log(softmax[y[i]])
        
        
        for l in range(C):
            
            if y[i] == l:
                dW[:,l] = dW[:,l] + (1 - softmax[l])*X[i]
            else:
                dW[:,l] = dW[:,l] - softmax[l]*X[i]
            
        # print(i, dW)
                
               
            
        
        
    loss = loss/N
    
    dW = - dW/N + 2* reg* W
    
    
    
    
    
    # print("./utils/classifiers/softmax.softmax_loss_naive() not implemented!") # delete me
    
    #############################################################################
    #                     END OF YOUR CODE                                      #
    #############################################################################


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    This adjusts the weights to minimize loss.

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
    #############################################################################
    #                     START OF YOUR CODE                                    #
    #############################################################################
    
    

    N = X.shape[0]
    D = X.shape[1]
    C= W.shape[1]

    # softmax 
    O = X.dot(W) # 100 20
    constant = np.max(O, axis=1).reshape(N,1)
    softmax_mat = np.exp(O - constant)                   # 100 20
    y_hat = softmax_mat/np.sum(softmax_mat, axis=1).reshape(N,1)  # 100 20  row sum = 1
    
    # loss
    # softmax = y_hat * Y
    cross_entropy = np.log(y_hat[range(N), y] ) # 100, 
    loss = -np.mean(cross_entropy)
    
    
    # y=y.reshape(len(y),1)
    Y = np.zeros((N,C))
    Y[range(N),y] = 1        # Y 100,20
    
    ###
#     for i in range(N):
        
#         dW= dW + ((Y[i,:] -y_hat[i,:]))*(np.array(X[i]).reshape(D,1))
      
#     dW = -dW/N + 2*reg*W
    ###
    
    # dW = -np.mean( (Y-y_hat)*X,axis=0 )+ 2*reg*W
    
    dW = -(np.dot(X.T, Y-y_hat))/N + 2*reg*W
    
    
    
    
#     dW1 = np.zeros((N,D,C))
#     for i in range(N):
        
#         dW1= dW1 + ((Y[i,:] -y_hat[i,:]))*(np.array(X).reshape(N, D,1))
    
#     # dW = dW1
    
#     dW = - np.mean(dW1, axis=0) + 2*reg*W
      
#     # dW = -dW/N + 2*reg*W
#     print(dW.shape)
    
    
    
    # print("./utils/classifiers/softmax.softmax_loss_vectorized() not implemented!") # delete me
    
    #############################################################################
    #                     END OF YOUR CODE                                      #
    #############################################################################
    

    return loss, dW
