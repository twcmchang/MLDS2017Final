import tensorflow as tf
import numpy as np
import tqdm
import os

## ----------------------
## Define Model_DNN class
## ----------------------
class Model_DNN(object):
    def __init__(self, model_name = 'DNN', n_steps = 40000,
                 input_size = 784, output_size = 10,
                 batch_size = 64, hidden_size_1 = 100, hidden_size_2 = 100,
                 act_fn = tf.nn.sigmoid, opt_fn = tf.train.GradientDescentOptimizer,
                 learning_rate = 0.01,
                 use_bn = True, momentum = 0.999, epsilon = 1e-3, is_training = True,
                 save_dir='save_DNN',
                 data=None,
                 record_every_n_steps = 50,
                 save_every_n_steps = 500):
        self.model_name = model_name
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.batch_size = batch_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.act_fn = act_fn
        self.opt_fn = opt_fn
        self.learning_rate = learning_rate
        self.use_bn = use_bn
        self.momentum = momentum
        self.epsilon = epsilon
        self.is_training = is_training
        self.save_dir = save_dir
        self.record_every_n_steps = record_every_n_steps
        self.save_every_n_steps = save_every_n_steps
        if data==None:
            print("Please specify the dataset in use")
        else:
            self.data = data
    
    def batch_norm_wrapper(self, inputs, is_training, momentum = 0.999, epsilon = 1e-3):
        scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
        beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
        pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
        pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

        if is_training:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])
            train_mean = tf.assign(pop_mean,
                                   pop_mean * momentum + batch_mean * (1 - momentum))
            train_var = tf.assign(pop_var,
                                  pop_var * momentum + batch_var * (1 - momentum))
            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(inputs,
                    batch_mean, batch_var, beta, scale, epsilon)
        else:
            return tf.nn.batch_normalization(inputs,
                pop_mean, pop_var, beta, scale, epsilon)
    
    def build_model(self, w1_init, w2_init, w3_init):
        # Placeholders
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_size])
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.output_size])
        
        # The first hidden layer
        self.w1 = tf.Variable(w1_init)
        if self.use_bn:
            self.z1 = tf.matmul(self.x,self.w1)
            self.bn1 = self.batch_norm_wrapper(self.z1, self.is_training, self.momentum, self.epsilon)
            self.l1 = self.act_fn(self.bn1)
        else:
            self.b1 = tf.Variable(tf.constant(0.1, shape=[self.hidden_size_1]))
            self.z1 = tf.matmul(self.x,self.w1)+self.b1
            self.l1 = self.act_fn(self.z1)
        
        # The second hidden layer
        self.w2 = tf.Variable(w2_init)
        if self.use_bn:
            self.z2 = tf.matmul(self.l1,self.w2)
            self.bn2 = self.batch_norm_wrapper(self.z2, self.is_training, self.momentum, self.epsilon)
            self.l2 = self.act_fn(self.bn2)
        else:
            self.b2 = tf.Variable(tf.constant(0.1, shape=[self.hidden_size_2]))
            self.z2 = tf.matmul(self.l1,self.w2)+self.b2
            self.l2 = self.act_fn(self.z2)

        # The output layer: always use softmax
        self.w3 = tf.Variable(w3_init)
        self.b3 = tf.Variable(tf.constant(0.1, shape=[self.output_size]))
        self.y  = tf.nn.softmax(tf.matmul(self.l2, self.w3)+self.b3)

        # Loss, Optimizer and Training operation
        self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y))
        self.train_step = self.opt_fn(self.learning_rate).minimize(self.cross_entropy)

        # Prediction
        self.correct_prediction = tf.equal(tf.arg_max(self.y,1),tf.arg_max(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32))

        # Gradients of loss w.r.t. weights
        self.w1_grad = tf.gradients(self.cross_entropy, [self.w1])[0]
        self.w2_grad = tf.gradients(self.cross_entropy, [self.w2])[0]
        self.w3_grad = tf.gradients(self.cross_entropy, [self.w3])[0]
        
        # Layer Jacobian
        self.l1_grad = tf.stack([tf.gradients(l1i, self.x)[0] for l1i in tf.unstack(self.l1, axis=1)],axis=2)
        self.l2_grad = tf.stack([tf.gradients(l1i, self.l1)[0] for l1i in tf.unstack(self.l2, axis=1)],axis=2)
        
        ## Keep the last 5 checkpoints
        self.saver = tf.train.Saver()
    
    def train(self, sess):
        self.w1_list, self.w1_grad_list = [], []
        self.w2_list, self.w2_grad_list = [], []
        self.accuracy_list, self.z2_list = [], []
        for idx in tqdm.tqdm(range(self.n_steps)):
            batch = self.data.train.next_batch(self.batch_size)
            _, self.w1_,  self.w2_, self.w1_grad_, self.w2_grad_ = sess.run([self.train_step, self.w1, self.w2, self.w1_grad, self.w2_grad],
                                                  feed_dict={self.x: batch[0], self.y_: batch[1]})
            self.w1_list.append(self.w1_)
            self.w2_list.append(self.w2_)
            self.w1_grad_list.append(self.w1_grad_)
            self.w2_grad_list.append(self.w2_grad_)
            if idx % self.record_every_n_steps is 0:
                self.accuracy_, self.z2_ = sess.run([self.accuracy, self.z2],
                                                    feed_dict={self.x: self.data.test.images,
                                                               self.y_: self.data.test.labels})
                self.accuracy_list.append(self.accuracy_)
                self.z2_list.append(np.mean(self.z2_, axis = 0))
            if idx % self.save_every_n_steps is 0:
                self.save(sess, self.save_dir, idx)

        self.w1_list = np.array(self.w1_list)
        self.w2_list = np.array(self.w2_list)
        self.w1_grad_list = np.array(self.w1_grad_list)
        self.w2_grad_list = np.array(self.w2_grad_list)
        self.accuracy_list = np.array(self.accuracy_list)
        self.z2_list = np.array(self.z2_list)

    def train_Jacob(self,sess,firstn=100):
        self.z1_list, self.z2_list, self.y_list = [],[],[]
        self.l1_grad_list, self.l2_grad_list, self.l3_grad_list = [],[],[]
        self.accuracy_list = []
        for idx in tqdm.tqdm(range(self.n_steps)):
            batch = self.data.train.next_batch(self.batch_size)
            _, self.l1_grad_, self.l2_grad_, self.l3_grad_ = sess.run([self.train_step,self.l1_grad, self.l2_grad, self.l3_grad],
                                                  feed_dict={self.x: batch[0], self.y_: batch[1]})
            if idx < firstn:
                self.l1_grad_list.append(self.l1_grad_)
                self.l2_grad_list.append(self.l2_grad_)
                self.l3_grad_list.append(self.l3_grad_)

            if idx % self.record_every_n_steps is 0:
                self.accuracy_, self.z1_, self.z2_, self.yy_ = sess.run([self.accuracy, self.z1, self.z2, self.y],
                                                    feed_dict={self.x: self.data.test.images,
                                                               self.y_: self.data.test.labels})
                self.accuracy_list.append(self.accuracy_)
                self.z1_list.append(np.mean(self.z1_, axis = 0))
                self.z2_list.append(np.mean(self.z2_, axis = 0))
                self.y_list.append(np.mean(self.yy_, axis = 0))

            if idx % self.save_every_n_steps is 0:
                self.save(sess, self.save_dir, idx)

        self.l1_grad_list = np.array(self.l1_grad_list)
        self.l2_grad_list = np.array(self.l2_grad_list)
        self.l3_grad_list = np.array(self.l3_grad_list)

        self.accuracy_list = np.array(self.accuracy_list)
        
        self.z1_list = np.array(self.z1_list)
        self.z2_list = np.array(self.z2_list)
        self.y_list = np.array(self.y_list)
    
    @property
    def model_dir(self):
        model_dir = "bsize{}_{}_{}_{}".format(self.batch_size,
                                           self.act_fn.__qualname__,
                                           self.opt_fn.__qualname__,
                                           str(self.learning_rate)[2:])
        if self.use_bn:
            model_dir = model_dir + '_BN'
        return model_dir
    
    def save(self, sess, save_dir, step):
        model_name = self.model_name + ".model"
        save_dir = os.path.join(save_dir, self.model_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.saver.save(sess, os.path.join(save_dir, model_name), global_step=step)
