from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine transformation function.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: a numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: a numpy array of weights, of shape (D, M)
    - b: a numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    """
    ############################################################################
    # TODO: Implement the affine forward pass. Store the result in 'out'. You  #
    # will need to reshape the input into rows.                                #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################
    # print(w.shape)
    
#     N = x.shape[0]
#     D = x.shape[1]
#     M = w.shape[1]
    # print(N,D,M)  # 100 784 100
    
    # out = np.zeros((N,M))
    # print(out)
    
    # print(b.shape）
          
    # x 100 784  N D
    # w 784 100  D M
    # b 100 1  M 1
    
    out = x.dot(w)+b
    
          

    # print('./utls/layer_funcs.affine_forward() not implemented!') # delete me
    
    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return out


def affine_backward(dout, x, w, b):
    """
    Computes the backward pass of an affine transformation function.

    Inputs:
    - dout: upstream derivative, of shape (N, M)
    - x: input data, of shape (N, d_1, ... d_k)
    - w: weights, of shape (D, M)
    - b: bias, of shape (M,)

    Returns a tuple of:
    - dx: gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: gradient with respect to w, of shape (D, M)
    - db: gradient with respect to b, of shape (M,)
    """
    ############################################################################
    # TODO: Implement the affine backward pass.                                #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################
    
    
    
    # dout N M;  w D M 
    dx = np.dot(dout,w.T)
    
    # dw D M  x = N D  dout N M
    dw = np.dot(x.T,dout )
    
    N=100
    ## not mean?
    # db = np.sum(dout,axis=1)
    # db = np.sum(dout,axis=0)/N
    db = np.sum(dout,axis=0)
    # db = np.mean(dout, axis=0)
    
    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for rectified linear units (ReLUs) activation function.

    Input:
    - x: inputs, of any shape

    Returns a tuple of:
    - out: output, of the same shape as x
    """
    ############################################################################
    # TODO: Implement the ReLU forward pass.                                   #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################
    
    # out = np.maximum(x, 0)
    # out = tuple(out)
    
    out = (x + np.abs(x))/2

    # print('./utls/layer_funcs.relu_forward() not implemented!') # delete me
    
    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return out


def relu_backward(dout, x):
    """
    Computes the backward pass for rectified linear units (ReLUs) activation function.

    Input:
    - dout: upstream derivatives, of any shape

    Returns:
    - dx: gradient with respect to x
    """
    ############################################################################
    # TODO: Implement the ReLU backward pass.                                  #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################
                 
    r = np.array(relu_forward(x))
    
    
    r[r<=0] = 0
    r[r>0] = 1
    
    # print("179", x.shape, relu_forward(x).shape,dout.shape, r.shape)
    
    dx = dout*r
    
    # print("layer.relu_backward return dx.shape ", dx.shape)             
    
    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return dx


def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    This adjusts the weights to minimize loss.
    y_prediction = argmax(softmax(x))

    Inputs:
    - x: (float) a tensor of shape (N, #classes)
    - y: (int) ground truth label, a array of length N

    Returns:
    - loss: the cross-entropy loss
    - dx: gradients wrt input x
    """
    # Initialize the loss.
    #loss = 0.0
    #dx = np.zeros_like(x)

    # When calculating the cross entropy,
    # you may meet another problem about numerical stability, log(0)
    # to avoid this, you can add a small number to it, log(0+epsilon)
    epsilon = 1e-15


    ############################################################################
    # TODO: You can use the previous softmax loss function here.               #
    # Hint: Be careful on overflow problem                                     #
    ############################################################################
    ############################################################################
    #                   START OF YOUR CODE                                     #
    ############################################################################
    
    
#     N = X.shape[0]
#     D = X.shape[1]
#     C= W.shape[1]

#     # softmax 
#     O = X.dot(W) # 100 20
#     constant = np.max(O, axis=1).reshape(N,1)
#     softmax_mat = np.exp(O - constant)                   # 100 20
#     y_hat = softmax_mat/np.sum(softmax_mat, axis=1).reshape(N,1)  # 100 20  row sum = 1

#     cross_entropy = np.log(y_hat[range(N), y] ) # 100, 
#     loss = -np.mean(cross_entropy)

#     Y = np.zeros((N,C))
#     Y[range(N),y] = 1        # Y 100,20

    
#     dW = -(np.dot(X.T, Y-y_hat))/N + 2*reg*W

    N = x.shape[0]
    C = x.shape[1]
    
    x_max = np.max(x, axis=1).reshape(N,1)
    softmax = np.exp(x-x_max)
    y_hat = softmax/np.sum(softmax, axis=1).reshape(N,1)
    
    cross_entropy = np.log(y_hat[range(N), y] ) 
    loss = -np.mean(cross_entropy)
    
    dx = np.zeros_like(x)
    
    Y = np.zeros((N,C))
    Y[range(N),y] = 1
    
    dx = (y_hat-Y)/N  # why deviding N?
#     print("o、x shape ", x.shape)
#     print("softmax_loss : dx.shape ",dx.shape )

    # print('./utls/layer_funcs.softmax_loss() not implemented!') # delete me
    
    ############################################################################
    #                    END OF YOUR CODE                                      #
    ############################################################################

    return loss, dx
