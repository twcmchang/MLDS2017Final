import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

import numpy as np
import tqdm
import os
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from model import Model_DNN

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


## ----------------------------------------------------
## Specify the directory to save checkpoint and figures
## ----------------------------------------------------
save_dir = 'save_Jacobian'

## -------------------------------------------------------------------------------
## Generate predetermined random weights so the networks are similarly initialized
## -------------------------------------------------------------------------------
w1_initial = tf.truncated_normal([784,100], stddev=np.sqrt(2 / 784), seed=5566)
w2_initial = tf.truncated_normal([100,100], stddev=np.sqrt(2 / 100), seed=5566)
w3_initial = tf.truncated_normal([100,10], stddev=np.sqrt(2 / 100), seed=5566)


## =========================================================
## (4) Example of Layer Jacobian
## Note: run 4000 steps only and compare (ReLU,sigmoid)
## =========================================================
act_fn_list = [tf.nn.sigmoid, tf.nn.relu]

w2_BN_dict, w2_grad_BN_dict, accuracy_BN_dict, z2_BN_dict = {}, {}, {}, {}
w2_dict, w2_grad_dict, accuracy_dict, z2_dict = {}, {}, {}, {}
l1_grad_BN_dict, l2_grad_BN_dict = {}, {}
l1_grad_dict, l2_grad_dict = {}, {}

for afn in act_fn_list:
    key = afn.__qualname__
    ## run model with BN:
    model = Model_DNN(n_steps = 1000, act_fn = afn, save_dir = save_dir, data=mnist)
    model.build_model(w1_initial,w2_initial,w3_initial)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.train_Jacob(sess, firstn=100)
    l1_grad_BN_dict[key] = model.l1_grad_list
    l2_grad_BN_dict[key] = model.l2_grad_list
    w2_BN_dict[key] = model.w2_list
    w2_grad_BN_dict[key] = model.w2_grad_list
    accuracy_BN_dict[key] = model.accuracy_list
    z2_BN_dict[key] = model.z2_list

    ## run model without BN:
    model = Model_DNN(n_steps = 1000, act_fn = afn, use_bn = False, save_dir = save_dir, data=mnist)
    model.build_model(w1_initial,w2_initial,w3_initial)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.train_Jacob(sess)

    l1_grad_dict[key] = model.l1_grad_list
    l2_grad_dict[key] = model.l2_grad_list
    w2_dict[key] = model.w2_list
    w2_grad_dict[key] = model.w2_grad_list
    accuracy_dict[key] = model.accuracy_list
    z2_dict[key] = model.z2_list

key_list = list(accuracy_dict.keys())
nlayer = [1,2]

for key in key_list:
    for k in nlayer: 
        if k == 1:
            l_grad = np.reshape(l1_grad_dict[key],newshape=[-1,l1_grad_dict[key].shape[2],l1_grad_dict[key].shape[3]])
            l_grad_BN = np.reshape(l1_grad_BN_dict[key],newshape=[-1,l1_grad_BN_dict[key].shape[2],l1_grad_BN_dict[key].shape[3]])
        else:
            l_grad = np.reshape(l2_grad_dict[key],newshape=[-1,l2_grad_dict[key].shape[2],l2_grad_dict[key].shape[3]])
            l_grad_BN = np.reshape(l2_grad_BN_dict[key],newshape=[-1,l2_grad_BN_dict[key].shape[2],l2_grad_BN_dict[key].shape[3]])

        from scipy.linalg import svdvals
        svd,svd_BN = [],[]
        for i in range(l_grad.shape[0]):
            svd.append(svdvals(l_grad[i]))
            svd_BN.append(svdvals(l_grad_BN[i]))

        svd = np.array(svd)
        svd_BN = np.array(svd_BN)

        plt.figure(figsize=(8,8))
        x1 = np.hstack(svd)
        x2 = np.hstack(svd_BN)

        data = np.vstack([x1,x2]).T
        bins = np.linspace(0.0, 1.0, 50)
        n,bins,patches = plt.hist(data,bins)
        plt.legend(loc='upper right',labels=['without BN','with BN'])
        plt.xlabel("Singluar value")
        plt.ylabel("Count")
        # plt.ylim([0,90000])
        plt.title("Singular Values of Layer "+str(k)+" Jacobian ("+key+",SGD")
        plt.savefig(os.path.join(save_dir,"hist_l"+str(k)+"_singular_"+key+"_SGD.png"))
        plt.show()
        plt.close()

        plt.figure(figsize=(8,8))
        x1 = np.hstack(svd)
        x2 = np.hstack(svd_BN)

        data = np.vstack([x1,x2]).T
        bins = np.linspace(0.5,1.0, 50)
        n,bins,patches = plt.hist(data,bins)
        plt.legend(loc='upper right',labels=['without BN','with BN'])
        plt.xlabel("Singluar value")
        plt.ylabel("Count")
        plt.title("Singular Values of Layer "+str(k)+" Jacobian ("+key+",SGD)")
        plt.savefig(os.path.join(save_dir,"hist_l"+str(k)+"_singular_"+key+"_SGD_closer.png"))
        plt.show()
        plt.close()

## ------------------------------------------------------
## [NOTE] Save results before running the next experiment
## ------------------------------------------------------