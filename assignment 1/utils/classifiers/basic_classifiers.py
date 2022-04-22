from __future__ import print_function

import numpy as np
from utils.classifiers.logistic_regression import *
from utils.classifiers.softmax import *


class BasicClassifier(object):
    def __init__(self):
        self.W = None
        self.velocity = None
    
    # Here, pass full training set (X_train and y_train) to train func
    
    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, optim='SGD', momentum=0.5, verbose=False):
        """
        Train this linear classifier using stochastic gradient descent(SGD).
        Batch size is set to 200, learning rate to 0.001, regularization rate to 0.00001.

        Inputs:
        - X: a numpy array of shape (N, D) containing training data; there are N
             training samples each of dimension D.
        - y: a numpy array of shape (N,) containing training labels; y[i] = c
             means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) L2 regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - optim: the optimization method, the default optimizer is 'SGD' and
                     feel free to add other optimizers.
        - verbose: (boolean) if true, print progress during optimization.

        Returns:
        - loss_history: a list containing the value of the loss function of each iteration.
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1  # assume y takes values 0...K-1 where K is number of classes

        # Initialize W and velocity(for SGD with momentum)
        # shape of W is (dim_of_x = X.shape[1], output = num_of_classes)
        if self.W is None:
            # proposed adjustment for W(D,1)
            if num_classes == 2:
                self.W = 0.001 * np.random.randn(dim, 1)
            else:
                self.W = 0.001 * np.random.randn(dim, num_classes)
                
       

        if self.velocity is None:
            self.velocity = np.zeros_like(self.W)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):

            #########################################################################
            # TODO:                                            #
            # Sample batch_size elements from the training data and their        #
            # corresponding labels to use in this round of gradient descent.      #
            # Store the data in X_batch and their corresponding labels in        #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)  #
            # and y_batch should have shape (batch_size,)                  #
            #                                                #
            # Hint: Use np.random.choice to generate indices. Sometimes, random    #
            # choice will be better than training in order.                 #
            #########################################################################
            #########################################################################
            #                     START OF YOUR CODE                                #
            #########################################################################
            
            
            indice = np.random.choice(num_train, batch_size, replace = False) # replacement 
            # print(indice)
            # update gradient of W using X[indice], y[indice]
            
            # print(X[indice].shape, y[indice].shape) # should be in (batch_size,dim) (batch_size,) 200, 785  200, 
            
            
            # print("./utils/classifiers/basic_classifiers.BasicClassifier.train() not implemented!") # delete me
            
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################
            
            #########################################################################
            # TODO:                                            #
            # Update the weights using the gradient and the learning rate.       #
            #########################################################################
            # evaluate loss and gradient
            #########################################################################
            #                     START OF YOUR CODE               #
            #########################################################################
            
            # import package has been done at the very first line!
            
            
#             loss, gradient = logistic_regression_loss_vectorized(self.W, X[indice], y[indice], reg)
#             # loss, dw_iter = logistic_regression_loss_naive(self.W, X[indice], y[indice], reg)
            
#             self.velocity = learning_rate*gradient
            
#             self.W = self.W - self.velocity
#             # print(self.W)
            
#             loss_history.append(loss)
# #             print(loss_history)
# #             print("\n")
#             # print('training accuracy: %f' % (np.mean(y == self.predict(X)) ))
    
            if num_classes == 2:
                # print( X[indice].shape)
                loss, gradient = logistic_regression_loss_vectorized(self.W, X[indice], y[indice], reg)
                self.velocity = learning_rate*gradient
                self.W = self.W - self.velocity
                loss_history.append(loss)
            else:
                loss, gradient = softmax_loss_vectorized(self.W, X[indice], y[indice], reg)
                self.velocity = learning_rate*gradient
                self.W = self.W - self.velocity
                loss_history.append(loss)
                
            #########################################################################
            #                    END OF YOUR CODE                  #
            #########################################################################
            
            # 当 达到100倍数时 report 一次
            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: a numpy array of shape (N, D) containing training data; there are N
             training samples each of dimension D.

        Returns:
        - y_pred: predicted labels for the data in X. y_pred is a 1-dimensional
                  array of length N, and each element is an integer giving the predicted
                  class.
        """
        #########################################################################
        # TODO:                                            #
        # Implement this method. Store the predicted labels in y_pred.       #
        #########################################################################
        #########################################################################
        #                     START OF YOUR CODE               #
        #########################################################################
        
        # parameter W is self.W
        
        # np.dot(X, self.W) output a (N, num_of_classes ) matrix, but in logistic regression
        # unlike softmax output full matrix, we just need one probability. This time  np.dot(X, self.W) output a (N, 1 ) ~  num_classes = 2
        
        
        
        if self.W.shape[1] == 1:
            probability = sigmoid(np.dot(X, self.W))
            y_pred = np.array(probability>0.5,dtype = np.int64 )
        else:
            # softmax 
            
            softmax_mat = np.exp(np.dot(X,self.W) )                   # 100 20
            y_hat = softmax_mat/np.sum(softmax_mat, axis=1).reshape(X.shape[0],1 )
            
            y_pred = np.array(np.argmax(y_hat,axis = 1) )
        
        # print("./utils/classifiers/basic_classifiers.BasicClassifier.predict() not implemented!") # delete me
        
        #########################################################################
        #                    END OF YOUR CODE                  #
        #########################################################################
        

        return y_pred
        
        
    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this, so no content needed for this function.

        Inputs:
        - X_batch: a numpy array of shape (N, D) containing a minibatch of N
                  data points; each point has dimension D.
        - y_batch: a numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns:
        - loss:  a single float
        - gradient:  gradients wst W, an array of the same shape as W
        """
        pass
        # pass, subclass will implement the details


class Logistic_Regression(BasicClassifier):
    """ A subclass that uses the Logistic Regression loss function """

    def loss(self, X_batch, y_batch, reg):
        return logistic_regression_loss_vectorized(self.W, X_batch, y_batch, reg)
    
    


class Softmax(BasicClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)
