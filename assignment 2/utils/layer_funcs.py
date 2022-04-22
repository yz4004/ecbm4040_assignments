#!/usr/bin/env/ python
# ECBM E4040 Fall 2021 Assignment 2
# This Python script contains various functions for layer construction.

import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)

    :return:
    - out: output, of shape (N, M)
    - cache: x, w, b for back-propagation
    """
    num_train = x.shape[0]
    x_flatten = x.reshape((num_train, -1))
    out = np.dot(x_flatten, w) + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.
    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
                    x: Input data, of shape (N, d_1, ... d_k)
                    w: Weights, of shape (D, M)

    :return: a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache

    N = x.shape[0]
    x_flatten = x.reshape((N, -1))

    dx = np.reshape(np.dot(dout, w.T), x.shape)
    dw = np.dot(x_flatten.T, dout)
    db = np.dot(np.ones((N,)), dout)

    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    :param x: Inputs, of any shape
    :return: A tuple of:
    - out: Output, of the same shape as x
    - cache: x for back-propagation
    """
    out = np.zeros_like(x)
    out[np.where(x > 0)] = x[np.where(x > 0)]

    cache = x

    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    :param dout: Upstream derivatives, of any shape
    :param cache: Input x, of same shape as dout

    :return: dx - Gradient with respect to x
    """
    x = cache

    dx = np.zeros_like(x)
    dx[np.where(x > 0)] = dout[np.where(x > 0)]

    return dx


def softmax_loss(x, y):
    """
    Softmax loss function, vectorized version.
    y_prediction = argmax(softmax(x))

    :param x: (float) a tensor of shape (N, #classes)
    :param y: (int) ground truth label, a array of length N

    :return: loss - the loss function
             dx - the gradient wrt x
    """
    loss = 0.0
    num_train = x.shape[0]

    x = x - np.max(x, axis=1, keepdims=True)
    x_exp = np.exp(x)
    loss -= np.sum(x[range(num_train), y])
    loss += np.sum(np.log(np.sum(x_exp, axis=1)))

    loss /= num_train

    neg = np.zeros_like(x)
    neg[range(num_train), y] = -1

    pos = (x_exp.T / np.sum(x_exp, axis=1)).T

    dx = (neg + pos) / num_train

    return loss, dx


def conv2d_forward(x, w, b, pad, stride):
    """
    A Numpy implementation of 2-D image convolution.
    By 'convolution', simple element-wise multiplication and summation will suffice.
    The border mode is 'valid' - Your convolution only happens when your input and your filter fully overlap.
    Another thing to remember is that in TensorFlow, 'padding' means border mode (VALID or SAME). For this practice,
    'pad' means the number rows/columns of zeroes to concatenate before/after the edge of input.

    Inputs:
    :param x: Input data. Should have size (batch, height, width, channels).
    :param w: Filter. Should have size (filter_height, filter_width, channels, num_of_filters).
    :param b: Bias term. Should have size (num_of_filters, ).
    :param pad: Integer. The number of zeroes to pad along the height and width axis.
    :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.

    :return: A 4-D array. Should have size (batch, new_height, new_width, num_of_filters).

    Note:
    To calculate the output shape of your convolution, you need the following equations:
    new_height = ((height - filter_height + 2 * pad) // stride) + 1
    new_width = ((width - filter_width + 2 * pad) // stride) + 1
    For reference, visit this website:
    https://adeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/
    """
    #######################################################################
    #                         TODO: YOUR CODE HERE                        #
    #######################################################################
    
    
    batch, height, width, channels = x.shape
    filter_height, filter_width, channels, num_of_filters = w.shape
    
    new_height = ((height - filter_height + 2 * pad) // stride) + 1
    new_width = ((width - filter_width + 2 * pad) // stride) + 1
    
    result = np.zeros((batch, new_height, new_width, num_of_filters))
    # print(batch,height,pad,width,channels)
    
    x_pad = np.zeros( (batch, height + 2*pad, width + 2*pad, channels) )
    # print(x_pad)
    x_pad[:,pad:x_pad.shape[1]-pad, pad:x_pad.shape[2]-pad  ,:] = x
    
    
    h_index = 0
    w_index = 0
    
    
    for batch_num in range(batch):
        
        for height in range(new_height):
            
            for width in range(new_width):
                
                for num in range(num_of_filters):
                    
                    conv_matrix = x_pad[batch_num, h_index:h_index+filter_height, w_index:w_index+filter_width,: ] * w[:,:,:,num]
                    
                    conv_result = np.sum(conv_matrix)
                    
                    result[batch_num, height, width, num] = conv_result + b[num]
              
                w_index += stride
            w_index = 0
            h_index += stride
        h_index = 0
        
    return result
    
    # print('./utils/layer_funcs.conv2d_forward() not implemented!') # delete me
    
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################


def conv2d_backward(d_top, x, w, b, pad, stride):
    """
    (Optional, but if you solve it correctly, we give you 5 points for this assignment.)
    A lite Numpy implementation of 2-D image convolution back-propagation.

    Inputs:
    :param d_top: The derivatives of pre-activation values from the previous layer
                       with shape (batch, height_new, width_new, num_of_filters).
    :param x: Input data. Should have size (batch, height, width, channels).
    :param w: Filter. Should have size (filter_height, filter_width, channels, num_of_filters).
    :param b: Bias term. Should have size (num_of_filters, ).
    :param pad: Integer. The number of zeroes to pad along the height and width axis.
    :param stride: Integer. The number of pixels to move between 2 neighboring receptive fields.

    :return: (d_w, d_b), i.e. the derivative with respect to w and b. For example, d_w means how a change of each value
     of weight w would affect the final loss function.

    Note:
    Normally we also need to compute d_x in order to pass the gradients down to lower layers, so this is merely a
    simplified version where we don't need to back-propagate.
    For reference, visit this website:
    http://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
    """
    #######################################################################
    #                         TODO: YOUR CODE HERE                        #
    #######################################################################
    
    # total_row = 
    
    
    
    
    # print('./utils/layer_funcs.conv2d_backward() not implemented!') # delete me
    
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    
def avg_pool_forward(x, pool_size, stride):
    """
    A Numpy implementation of 2-D image average pooling.

    Inputs:
    :params x: Input data. Should have size (batch, height, width, channels).
    :params pool_size: Integer. The size of a window in which you will perform average operations.
    :params stride: Integer. The number of pixels to move between 2 neighboring receptive fields.
    :return :A 4-D array. Should have size (batch, new_height, new_width, num_of_filters).
    """
    #######################################################################
    #                         TODO: YOUR CODE HERE                        #
    #######################################################################
    
    
    batch, height, width, channels = x.shape
    
    num_of_filters = channels
    
    new_height = (height - pool_size)  // stride + 1
    new_width = (width - pool_size) // stride + 1
    
    result = np.zeros((batch, new_height, new_width, num_of_filters))
    
    h_index = 0
    w_index = 0
    
    for num_batch in range(batch):
        
        for num_height in range(new_height):
            
            for num_width in range(new_width):
                
                for num_of_filter in range(num_of_filters):
                    
                    # print(h_index,w_index)
                    # print(num_height,num_width)
                    # print(x[num_batch, h_index:h_index+pool_size  ,w_index:w_index+pool_size, num_of_filter ])
                    
                    pool_result = np.mean(x[num_batch, h_index:h_index+pool_size ,w_index:w_index+pool_size,num_of_filter ])
                    
                    result[num_batch,num_height,num_width,num_of_filter] = pool_result
                    
                w_index += stride
            w_index=0
            h_index+=stride
        h_index = 0
    
    
    return result
    
    # print('./utils/layer_funcs.avg_pool_forward() not implemented!') # delete me
    
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################

    
def avg_pool_backward(dout, x, pool_size, stride):
    """
    (Optional, but if you solve it correctly, we give you +5 points for this assignment.)
    A Numpy implementation of 2-D image average pooling back-propagation.

    Inputs:
    :params dout: The derivatives of values from the previous layer
                       with shape (batch, height_new, width_new, num_of_filters).
    :params x: Input data. Should have size (batch, height, width, channels).
    :params pool_size: Integer. The size of a window in which you will perform average operations.
    :params stride: Integer. The number of pixels to move between 2 neighboring receptive fields.
    
    :return dx: The derivative with respect to x
    You may find this website helpful:
    https://medium.com/the-bioinformatics-press/only-numpy-understanding-back-propagation-for-max-pooling-layer-in-multi-layer-cnn-with-example-f7be891ee4b4
    """
    #######################################################################
    #                         TODO: YOUR CODE HERE                        #
    #######################################################################
    
    index_h, index_w, index_c = (0,0,0)
    dx = np.zeros_like(x,dtype = np.float32)
    # dout = result

    

    batchs, height, width, channels = x.shape
    

    batch_dout, height_dout, width_dout, channels_dout = dout.shape

    # stride = 1
    # pool_size = 2

    
    # new_height = ((height -pool_size ) // stride) + 1
    # new_width = ((width - pool_size) // stride) + 1


#     pad = pool_size-1
#     dout_pad =  np.zeros( (batch_dout, height_dout + 2*pad, width_dout + 2*pad, channels_dout) )
#     dout_pad[:, pad:dout_pad.shape[1]-pad,  pad:dout_pad.shape[2]-pad  ,:] = dout
    
    
    bth = 0
    btw = 0
    
    for ind_batch in range(batchs):
        
        for ind_h in range(0,x.shape[1]-x.shape[1]%stride,stride):
    
            for ind_w in range(0,x.shape[2]-x.shape[2]%stride,stride):
            
                # print(ind_h,ind_w)

        
                for ind_c in range(x.shape[3]):
                    
                    # matrix_in_dout = dout_pad[ind_batch ,btw:btw+pool_size, bth:bth+pool_size, ind_c]
                    for s1 in range(pool_size):
                        for s2 in range(pool_size):
                            dl_dy = dout[ind_batch ,bth, btw, ind_c]
                            dx[ind_batch, ind_h+s1, ind_w+s2,ind_c] = dl_dy/pool_size**2
                    
                    
                
            
            # btw += pool_size # stride
                btw += 1
            
            btw = 0
        # bth += pool_size# stride
            bth += 1
        bth = 0
    
    return(dx)
    # print('./utils/layer_funcs.avg_pool_backward() not implemented!') # delete me
    
    #######################################################################
    #                           END OF YOUR CODE                          #
    #######################################################################
