import tensorflow as tf

def LSTM_step(cell_inputs, cell_states, kernel, recurrent_kernel, bias):
    """
    Run one time step of the cell. That is, given the current inputs and the cell states from the last time step, calculate the current state and cell output.
    You will notice that TensorFlow LSTMCell has a lot of other features. But we will not try them. Focus on the very basic LSTM functionality.
    Hint: In LSTM there exist both matrix multiplication and element-wise multiplication. Try not to mix them.
        
        
    :param cell_inputs: The input at the current time step. The last dimension of it should be 1.
    :param cell_states:  The state value of the cell from the last time step, containing previous hidden state h_tml and cell state c_tml.
    :param kernel: The kernel matrix for the multiplication with cell_inputs
    :param recurrent_kernel: The kernel matrix for the multiplication with hidden state h_tml
    :param bias: Common bias value
    
    
    :return: current hidden state, and a list of hidden state and cell state. For details check TensorFlow LSTMCell class.
    """
    
    
    ###################################################
    # TODO:      INSERT YOUR CODE BELOW               #
    # params                                          #
    ###################################################
    
    # tf.zeros_like()
    
    # cell_inputs (d,1), x_t 
    # kernel (d,h)
    # recurrent_kernel (h,h)
    # cell_states = (hidden_state, cell_state) = (h_tml, c_tml)
    
    # cell_inputs (d,1), x_t 
    X = cell_inputs  
    # X = tf.cast(X, tf.float32)
    X = tf.reshape(X,[-1,kernel.shape[0]])
    
    # cell_states = (hidden_state, cell_state) = (h_tml, c_tml)  H: (h,) C: (c, )
    # H, C = cell_states  # tuple
    H = cell_states[0]   # list 
    C = cell_states[1] 
    
    
    # h
    h = H.shape[1]
    
    # Forget Gate __________
    # (Wfh, Wfx) bf
    Wfh, Wfx, bf = recurrent_kernel[:,:h],kernel[:,:h], bias[:h]
    # print(Wfh.shape, Wfx.shape,bf.shape)
    F = tf.sigmoid( tf.matmul(H,Wfh) + tf.matmul(X, Wfx) + bf)
    
    # Input Gate & candidate cell _________
    # (Wih, Wix) bi
    # (Wch, Wcx) bc
    Wih, Wix, bi = recurrent_kernel[:,h:2*h],kernel[:,h:2*h], bias[h:2*h]
    # print(Wih.shape)
    Wch, Wcx, bc = recurrent_kernel[:,2*h:3*h],kernel[:,2*h:3*h], bias[2*h:3*h]
    # print(Wch.shape)
    I = tf.sigmoid( tf.matmul(H,Wih) + tf.matmul(X, Wix) + bi ) 
    C_tilde = tf.tanh( tf.matmul(H,Wch) + tf.matmul(X, Wcx) + bc )
    
    # cell state / cel memory_________
    # print(F.shape, C.shape, I.shape, C_tilde.shape)
    C = F*C + I*C_tilde
    
    # Output Gate & Hidden State ______
    # (Woh, Wox) bo
    Woh, Wox, bo = recurrent_kernel[:,3*h:],kernel[:,3*h:], bias[3*h]
    O = tf.sigmoid( tf.matmul(H,Woh) + tf.matmul(X, Wox) + bo  )
    H = O*tf.tanh(C)
    
    return O, [H,C]
    
    
    # return: current hidden state, and a list of hidden state and cell state. For details check TensorFlow LSTMCell class.
    
    ###################################################
    # END TODO                                        #
    ###################################################
