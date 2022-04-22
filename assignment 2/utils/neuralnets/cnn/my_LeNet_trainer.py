#!/usr/bin/env/ python
# ECBM E4040 Fall 2020 Assignment 2
# TensorFlow custom CNN model
import tensorflow as tf
from utils.neuralnets.cnn.model_LeNet import LeNet
from utils.image_generator import ImageGenerator
from tqdm import tqdm


class MyLeNet_trainer():
    """
    X_train: Train Images. It should be a 4D array like (n_train_images, img_len, img_len, channel_num).
    y_train: Train Labels. It should be a 1D vector like (n_train_images, )
    X_val: Validation Images. It should be a 4D array like (n_val_images, img_len, img_len, channel_num).
    y_val: Validation Labels. It should be a 1D vector like (n_val_images, )
    epochs: Number of training epochs
    batch_size: batch_size while training
    lr: learning rate of optimizer
    """
    def __init__(self,X_train, y_train, X_val, y_val, epochs=10, batch_size=256, lr=1e-3):
        self.X_train = X_train.astype("float32")
        self.y_train = y_train.astype("float32")
        self.X_val = X_val.astype("float32")
        self.y_val = y_val.astype("float32")
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

    # Initialize MyLenet model
    def init_model(self):
        self.model = LeNet(self.X_train[0].shape)

    #initialize loss function and metrics to track over training
    def init_loss(self):
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    # Initialize optimizer
    def init_optimizer(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

    # Prepare batches of train data using ImageGenerator
    def batch_train_data(self, batch_size=256, shuffle=True):
        train_data = ImageGenerator(self.X_train, self.y_train)
        #############################################################
        # TODO: create augumented data with your proposed data augmentations
        #       from part 3
        #############################################################
        
        # gen = ImageGenerator(X_dev.reshape(-1,1,28,28).transpose(0,2,3,1), y_dev)
        # gen.show(X_dev)
        train_data = ImageGenerator(X_train, y_train)
        # X_train: (n_train_images, img_len, img_len, channel_num)
        
        
        # flip("h")
        # flip("hv")
        # brightness("1.5")
        # add_noise(0.2,1) # 20% pictures add noise level 
        # translate(3,-3)
        
        bright = gen.brightness(1.5)
        flipped1 = gen.flip('h')
        flipped2 = gen.flip('hv')
        translated = gen.translate(3,-3)
        noised = gen.add_noise(1,1) 
        
        train_data.create_aug_data()
        
        
        # data_iter = train_data.next_batch_gen(batch_size, shuffle)
        # features, labels = next(iter(data_iter))
        
        # return features, labels
        
        # print('./utils/neuralnets/cnn/my_LeNet_trainer.MyLeNet_trainer.batch_train_data() not implemeted!') # delete me
        
        #############################################################
        # END TODO
        #############################################################
        
        self.train_data_next_batch = train_data.next_batch_gen(batch_size,shuffle=shuffle)
        self.n_batches = train_data.N_aug // batch_size
    
    # Define training step
    def train_step(self, images, labels, training=True):
        with tf.GradientTape() as tape:
        # training=True is always recommended as there are few layers with different
        # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=training)
            loss = self.loss_function(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    # Define testing step
    def test_step(self, images, labels, training=False):
        predictions = self.model(images, training=training)
        t_loss = self.loss_function(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    # train epoch
    def train_epoch(self, epoch):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_accuracy.reset_states()
        #############################################################
        # TODO: use data from ImageGenerator to train the network
        # hint: use python next feature "next(self.train_data_next_batch)""
        #############################################################
        
        data_iter = gen.next_batch_gen(batch_size, shuffle)
        # features, labels = next(iter(data_iter))
        
        for X,y in data_iter:
            # X,y = next(data_iter)
            self.train_step(X,y,training = True)
        
        # ?
        with self.train_summary_writer.as_default():
            tf.summary.scalar('loss', self.train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', self.train_accuracy.result(), step=epoch)

        for test_images, test_labels in test_ds:
            self.test_step(test_images, test_labels)
            
        with self.test_summary_writer.as_default():
            tf.summary.scalar('loss', self.test_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', self.test_accuracy.result(), step=epoch)
        
        
        
        
        # print('./utils/neuralnets/cnn/my_LeNet_trainer.MyLeNet_trainer.train_epoch() not implemeted!') # delete me
        
        #############################################################
        # END TODO
        #############################################################


        self.test_step(self.X_val, self.y_val)

        template = 'Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
        print(template.format(self.train_loss.result(),
                            self.train_accuracy.result() * 100,
                            self.test_loss.result(),
                            self.test_accuracy.result() * 100))
            
    # start training
    def run(self):
        self.init_model()
        self.init_loss()
        self.init_optimizer()
        self.batch_train_data()

        for epoch in range(self.epochs):
            print('Training Epoch {}'.format(epoch + 1))
            self.train_epoch(epoch)
    
