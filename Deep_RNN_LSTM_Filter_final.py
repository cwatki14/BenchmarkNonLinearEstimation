from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math

#Sets RNN model parameters
num_epochs = 10000**1
series_length = 100
truncated_backprop_length = series_length
truncated_backprop_length = 50
state_size = int(32*5)
batch_size = 10   # should be odd if less than 6
num_batches = series_length//truncated_backprop_length
num_layers = 3
#for constant learning rate
#learning_rate = 1e-4
tf.reset_default_graph() 
nm = 1
ns = 1
#%%
def generateData(batch_size,series_length):
    """Generates training and testing data via simulation
    x_t is the Desired Output
    y_t is the Measured Output
    Measured inputs are determined by the specific example number
    Neural net Inputs: Measured Inputs, Measured Outputs, Desired Outputs (only during training)
    Neural net Outputs: Desired Outputs   
    Minimize some desired performance metric (e.g, mean square error)
    """
    
    #x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    x = np.zeros((batch_size,series_length, ns))
    y = np.zeros((batch_size,series_length, nm))
    for curr_batch in range(batch_size):
        #Commented blocks are for other benchmarks that could be considered
        
        # Example 1 of J. T. Lo (optimal RMSE around 0.211)
#        x_t = -0.5 + 0.1*np.random.randn()
#        y_t = x_t**3
#        for t in range(series_length):
#            w_t = 1*np.random.randn()
#            v_t = 1*np.random.randn()
#            x[curr_batch,t] = x_t
#            y[curr_batch,t] = y_t
#            x_t = 1.1*math.e**(-2*(x_t**2)) - 1 + 0.5*w_t
#            y_t = x_t**3 + 0.1*v_t

        # Example 2 of J. T. Lo (optimal RMSE around 0.18)
#        x_t = 0 + 0.5*np.random.randn()
#        y_t = x_t**3
#        for t in range(series_length):
#            w_t = 1*np.random.randn()
#            v_t = 1*np.random.randn()
#            x[curr_batch,t] = x_t
#            y[curr_batch,t] = y_t
#            x_t = 1.7*math.e**(-2*(x_t**2)) - 1 + 0.1*w_t
#            y_t = x_t**3 + 0.1*v_t

        # Example 3 of J. T. Lo (optimal RMSE around 0.35)
#        x_t = 0 + 1*np.random.randn()
#        y_t = x_t
#        for t in range(series_length):
#            w_t = 1*np.random.randn()
#            v_t = 1*np.random.randn()
#            x[curr_batch,t] = x_t
#            y[curr_batch,t] = y_t
#            x_t = 0.9*x_t + 0.2*w_t
#            y_t = x_t + v_t

        # Example 4 of J. T. Lo (optimal RMSE around 0.056)
#        x_t = 0 + 0.5*np.random.randn()
#        y_t = x_t
#        for t in range(series_length):
#            w_t = 1*np.random.randn()
#            v_t = 1*np.random.randn()
#            x[curr_batch,t] = x_t
#            y[curr_batch,t] = y_t
#            x_t = 0.5*x_t + 0.5*math.tanh(x_t + 0.5*w_t)
#            y_t = x_t + 0.5*(x_t**3)*v_t

        # Time Series Benchmark
        sigma_x = 1*math.sqrt(25)  # 10
        sigma_u = 1*math.sqrt(10)             # 10
        sigma_v = 1*math.sqrt(1)
        x_t = 0.1 + 1.01*sigma_x*np.random.randn()
        y_t = (1/20)*x_t**2
        for t in range(series_length):
            u_t = 1.0*sigma_u*np.random.randn()  # nominally, math.sqrt(10)*np.random.randn()
            v_t = 1*sigma_v*np.random.randn()      # nominally, np.random.randn() 
            x[curr_batch,t] = x_t
            y[curr_batch,t] = y_t
            x_t = 0.5*x_t + 25*x_t/(1+x_t**2) + 8*math.cos(1.2*t) + u_t
            y_t = (1/20)*x_t**2 + 1*v_t

        # Simple Second-order Markov Process
#        tau = 0.1
#        wn = 6
#        zeta = 0.16
#        Qd = 1e-2
#        Qk = np.matrix([[0, 0], [0, Qd] ])
#        sigma_r = 1e-2
#        Phi = np.array([[1, 0],[0, 1]]) + tau*np.array([[0, 1],[-wn**2, -2*zeta*wn]])
#        H = np.array([1, 0])
##        H = np.identity(nm)
#        x_t = np.array([[np.random.randn(), np.random.randn()]])
#        y_t = np.matmul(H,x_t.T).T
#        for t in range(series_length):
#            v_t = np.random.multivariate_normal([0, 0], Qk, 1)
#            x[curr_batch,t,:] = x_t[0]
#            y[curr_batch,t,:] = y_t[0]
#            x_t = np.matmul(Phi,x_t.T).T + v_t
#            w_t = sigma_r*np.random.randn()  # nominally, math.sqrt(10)*np.random.randn()
#            y_t = np.matmul(H,x_t.T).T + w_t

        # Ballistic Object
#        tau = 0.1
#        q1 = 1*5
#        q2 = 1*5
##        Qk = np.matrix([[(q1/3)*tau**3, (q1/2)*tau**2, 0], [(q1/2)*tau**2, q1*tau, 0], [0, 0, q2*tau] ])
#        Qk = np.matrix([[(q1/3)*tau**3, (q1/2)*tau**2], [(q1/2)*tau**2, q1*tau] ])
#        sigma_r = 200*1e-3
#        Phi = np.array([[1, 0],[0, 1]]) + tau*np.array([[0, -1],[0, 0]])
##        Phi = np.array([[1, -tau, 0],[0, 1, 0],[0, 0, 1]])
#        H = np.array([1, 0,])
#        #H = np.identity(3)
#        g = 9.81*1e-3
#        x_t = np.array([[61 + np.random.randn(), 3.048 + np.random.randn()]])
#        y_t = np.matmul(H,x_t.T).T
#        for t in range(series_length):
#            v_t = 1e-3*np.random.multivariate_normal([0, 0], Qk, 1)
#            x[curr_batch,t,:] = x_t[0]
#            y[curr_batch,t,:] = y_t[0]
#            x_t = np.matmul(Phi,x_t.T).T + tau*np.array([0, g]).T + v_t
#            w_t = sigma_r*np.random.randn()  # nominally, math.sqrt(10)*np.random.randn()
#            y_t = np.matmul(H,x_t.T).T + w_t

    return (x, y)

#%%
#Builds RNN model in tensorflow
batchX_placeholder =  tf.placeholder(tf.float32, [batch_size, truncated_backprop_length, nm])
batchY_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length, ns])

init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])

state_per_layer_list = tf.unstack(init_state, axis=0)
rnn_tuple_state = tuple(
    [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
     for idx in range(num_layers)]
)

W2 = tf.Variable(np.random.rand(state_size, ns),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1, ns)), dtype=tf.float32)

#%%
# Forward pass
stacked_rnn = []
for iiLyr in range(num_layers):
    stacked_rnn.append(tf.nn.rnn_cell.LSTMCell(state_size, state_is_tuple=True))

multiLyr_cell = tf.nn.rnn_cell.MultiRNNCell(stacked_rnn, state_is_tuple=True)
states_series, current_state = tf.nn.dynamic_rnn(cell=multiLyr_cell, inputs=batchX_placeholder, initial_state=rnn_tuple_state)
states_series = tf.reshape(states_series, [-1, state_size])

#%%
predictions = tf.matmul(states_series, W2) + b2 #Broadcasted addition
labels = tf.reshape(batchY_placeholder, [truncated_backprop_length*batch_size, ns])
predictions_series = tf.unstack(tf.reshape(predictions, [batch_size, truncated_backprop_length, ns]), axis=1)

labels_series = tf.unstack(tf.reshape(labels, [batch_size, truncated_backprop_length, ns]), axis=1)

losses = [tf.pow(pred - label, 2) for pred, label in zip(predictions_series,labels_series)]
#For MAE instead of RSME
#losses_l1 = [tf.abs(tf.pow(pred - label, 1)) for pred, label in zip(predictions_series,labels_series)]
total_loss = tf.reduce_mean(losses)
#total_loss_l1 = tf.reduce_mean(losses_l1)

#For learning rate decay
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1e-3
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000, 0.96, staircase=True)
#If MAE is selected
#train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss_l1)
#If RSME is selected
train_step = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
#%%
def plot(loss_list, predictions_series, batchX, batchY):
    """Creates training plots"""
    fig = plt.figure('Performance', figsize=(10,10), facecolor=[0.7,0.7,0.7], edgecolor='k')
    total_subplot = min(batch_size + 1,2)
    ax1 = fig.add_subplot(total_subplot, 1, 1)
    ax1.cla()
    ax1.plot(loss_list)
    ax1.set_ylabel('Cost Function', color = 'k', fontsize=14, fontweight = 'semibold')
    ax1.set_xlabel('Iteration', color = 'k', fontsize=14, fontweight = 'semibold')
    ax1.tick_params(labelsize=14)
    
    for batch_series_idx in range(total_subplot-1):
        single_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        
        ax = fig.add_subplot(total_subplot, 1, batch_series_idx + 2)
        ax.cla()
        #plt.axis([0, truncated_backprop_length, -2, 2])
        ax.plot(batchX[batch_series_idx, :,0], linewidth=1, color="magenta")    # measured output = input to RNN
        ax.plot(batchY[batch_series_idx, :,:], linewidth=2, color="red")     # state = desired output
        ax.plot(single_output_series, linewidth=2, color="blue")
        ax.set_xlabel('Time (seconds', color = 'k', fontsize=14, fontweight = 'semibold')
        ax.set_ylabel('Signals', color = 'k', fontsize=14, fontweight = 'semibold')
        ax.tick_params(labelsize=14)
    
    plt.draw()
    plt.pause(0.0001)

def plot_validation(loss_list, predictions_series, batchX, batchY):
    """Creates collection of validation subplots"""
    fig = plt.figure('Performance', figsize=(10,10), facecolor=[0.7,0.7,0.7], edgecolor='k')
    total_subplot = min(batch_size + 1,10)
    ax1 = fig.add_subplot(total_subplot, 1, 1)
    ax1.cla()
    ax1.plot(loss_list)
    #ax1.set_ylabel('Cost Function', color = 'k', fontsize=14, fontweight = 'semibold')
    #ax1.set_xlabel('Iteration', color = 'k', fontsize=14, fontweight = 'semibold')
    ax1.tick_params(labelsize=14)
    
    for batch_series_idx in range(total_subplot-1):
        single_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        
        ax = fig.add_subplot(total_subplot, 1, batch_series_idx + 2)
        ax.cla()
        #plt.axis([0, truncated_backprop_length, -2, 2])
        ax.plot(batchX[batch_series_idx, :,0], linewidth=1, color="magenta")    # measured output = input to RNN
        ax.plot(batchY[batch_series_idx, :,:], linewidth=2, color="red")     # state = desired output
        ax.plot(single_output_series, linewidth=2, color="blue")
        #ax.set_xlabel('Time (seconds', color = 'k', fontsize=14, fontweight = 'semibold')
        #ax.set_ylabel('Signals', color = 'k', fontsize=14, fontweight = 'semibold')
        ax.tick_params(labelsize=14)
    
    plt.draw()
    plt.pause(0.000001)

saver = tf.train.Saver()

#%%
#Executes n-layer RNN filter training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []
    best_loss = 1000
    curr_loss = best_loss
#    saver.restore(sess, "/tmp/model1.ckpt")
    print("----------- Training Run(s) ----------")  
    for epoch_idx in range(num_epochs):
        x,y = generateData(batch_size,series_length)
        _current_state = np.zeros((num_layers, 2, batch_size, state_size))
        
        print("------------ Epoch number ------------", epoch_idx)
                            
        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = y[:,start_idx:end_idx,:]     # inputs to Neural Net = measured output
            batchY = x[:,start_idx:end_idx,:]     # desired outputs of Neural Net = states
            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                    [total_loss, train_step, current_state, predictions_series],
                    feed_dict={
                        batchX_placeholder: batchX,
                        batchY_placeholder: batchY,
                        init_state: _current_state
                    })
            
        change_loss = math.sqrt(_total_loss) - math.sqrt(curr_loss)
        curr_loss = _total_loss
        
        best_loss = min(best_loss, curr_loss)
                
        print("Loss after epoch number ",epoch_idx, ' = ', math.sqrt(curr_loss))
        print("Loss function improvement = ", change_loss)  # must be positive
        print("-----------------------------------")
        loss_list.append(math.sqrt(curr_loss))
        if (epoch_idx % 200 == 0):
            plot(loss_list, _predictions_series, batchX, batchY)
            plt.ioff()
            plt.show()

    print("Average loss (last 100 samples) during training = ", np.mean(loss_list[-100:-1]))
    plot(loss_list, _predictions_series, batchX, batchY)
    plt.ioff()
    plt.show()

    save_path = saver.save(sess, "/tmp/model1.ckpt")

#% Validation run
    print("----------- Validation Run(s) ----------")    
    loss_list_v = []
    #saver.restore(sess, "/tmp/model.ckpt")
    for epoch_idx in range(10):
        x,y = generateData(batch_size,series_length)
        _current_state = np.zeros((num_layers, 2, batch_size, state_size))
        
        print("Epoch number ", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = y[:,start_idx:end_idx]
            batchY = x[:,start_idx:end_idx]
            _total_loss, _current_state, _predictions_series = sess.run(
                [total_loss, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder: batchX,
                    batchY_placeholder: batchY,
                    init_state: _current_state
                })
    
        print("Loss", math.sqrt(_total_loss))
        loss_list_v.append(math.sqrt(_total_loss))

    print("Average loss during validation = ", np.mean(loss_list_v))
    plot_validation(loss_list_v, _predictions_series, batchX, batchY)
    plt.ioff()
    plt.show()
      