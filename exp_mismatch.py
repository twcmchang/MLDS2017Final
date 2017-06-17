import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

# loading data
import numpy as np
import tqdm
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
ori_mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

from model import Model_DNN
from util import DataSet, DataSets

global mnist
mnist = DataSets()

# balanced training dataset
i1 = np.where((ori_mnist.train.labels[:,1]==1))[0]
i2 = np.where((ori_mnist.train.labels[:,0]==1))[0]
np.random.shuffle(i1)
np.random.shuffle(i2)
mylen = 5000
i1 = i1[:mylen]
i2 = i2[:mylen]
itrain = np.append(i1,i2)
np.random.shuffle(itrain)

train_img = np.array([ori_mnist.train.images[j] for j in itrain])
train_lab = np.array([ori_mnist.train.labels[j] for j in itrain])
mnist.train = DataSet(train_img,train_lab)

## ----------------------------------------------------
## Specify the directory to save checkpoint and figures
## ----------------------------------------------------
save_dir = 'save_exp_mismatch'

## -------------------------------------------------------------------------------
## Generate predetermined random weights so the networks are similarly initialized
## -------------------------------------------------------------------------------
w1_initial = tf.truncated_normal([784,100], stddev=np.sqrt(2 / 784), seed=5566)
w2_initial = tf.truncated_normal([100,100], stddev=np.sqrt(2 / 100), seed=5566)
w3_initial = tf.truncated_normal([100,10], stddev=np.sqrt(2 / 100), seed=5566)

## =========================================================
## (4) Try different distribution of testing data
## =========================================================

pratio_list = [0.5, 0.4, 0.3, 0.2, 0.1]
w2_BN_dict, w2_grad_BN_dict, accuracy_BN_dict, z2_BN_dict = {}, {}, {}, {}
w2_dict, w2_grad_dict, accuracy_dict, z2_dict = {}, {}, {}, {}

for rat in pratio_list:
    # get testing dataset
    i1 = np.where((ori_mnist.test.labels[:,1]==1))[0]
    i2 = np.where((ori_mnist.test.labels[:,0]==1))[0]
    np.random.shuffle(i1)
    np.random.shuffle(i2)
    mylen_t = 1000
    i1 = i1[:int(mylen*rat)]
    i2 = i2[:int(mylen*(1-rat))]
    itest = np.append(i1,i2)
    np.random.shuffle(itest)
    test_img = np.array([ori_mnist.test.images[j] for j in itest])
    test_lab = np.array([ori_mnist.test.labels[j] for j in itest])
    mnist.test = DataSet(test_img,test_lab)
    
    key = str(rat*100)
    ## run model with BN:
    model = Model_DNN(n_steps = 40000, save_dir = save_dir, data=mnist)
    model.build_model(w1_initial,w2_initial,w3_initial)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.train(sess)
    w2_BN_dict[key] = model.w2_list
    w2_grad_BN_dict[key] = model.w2_grad_list
    accuracy_BN_dict[key] = model.accuracy_list
    z2_BN_dict[key] = model.z2_list
    ## run model without BN:
    model = Model_DNN(n_steps = 40000, use_bn = False, save_dir = save_dir, data=mnist)
    model.build_model(w1_initial,w2_initial,w3_initial)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.train(sess)
    w2_dict[key] = model.w2_list
    w2_grad_dict[key] = model.w2_grad_list
    accuracy_dict[key] = model.accuracy_list
    z2_dict[key] = model.z2_list

## =======================
## plot w.r.t. mismatch
## =======================

fig, ax = plt.subplots(figsize=(12,8))
color_list = ['b', 'g', 'r', 'k', 'c', 'm']

# keys in dictionary
key_list = [str(x*100) for x in pratio_list]

## (sort by accuracy in decreasing order)
final_accuracy_BN_dict = [accuracy_BN_dict[i][-1] for i in key_list]
decreasing_order = sorted(range(len(final_accuracy_BN_dict)),
                          key=lambda k: -final_accuracy_BN_dict[k])

## Plot accuracy with BN
for idx in decreasing_order:
    ax.plot(range(0,len(accuracy_BN_dict[key_list[idx]])*model.record_every_n_steps,model.record_every_n_steps),
            accuracy_BN_dict[key_list[idx]],
            color = color_list[idx], alpha = 0.4, linewidth = 2.5,
            label='pos:neg= %s:%s BN'%(int(100*((1-pratio_list[idx])/pratio_list[idx])),int(1)))

## Plot accuracy without BN
for idx in decreasing_order:
    ax.plot(range(0,len(accuracy_dict[key_list[idx]])*model.record_every_n_steps,model.record_every_n_steps),
            accuracy_dict[key_list[idx]],
            color = color_list[idx], alpha = 0.8, linewidth = 1.2,
            linestyle = '--',
            label='pos:neg= %s:%s BN'%(int(100*((1-pratio_list[idx])/pratio_list[idx])),int(1)))

ax.set_xlabel('Training steps')
ax.set_ylabel('Accuracy')
ax.set_title('Mismatch between Training/Testing')
ax.legend(loc=4)
plt.savefig(os.path.join(save_dir, 'accuracy_vs_mismatch.png'))
plt.show()
plt.close()
## ------------------------------------------------------
## [NOTE] Save results before running the next experiment
## ------------------------------------------------------