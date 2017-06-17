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
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

from model import Model_DNN

## ----------------------------------------------------
## Specify the directory to save checkpoint and figures
## ----------------------------------------------------
save_dir = 'save_exp_actfn'

## -------------------------------------------------------------------------------
## Generate predetermined random weights so the networks are similarly initialized
## -------------------------------------------------------------------------------
w1_initial = tf.truncated_normal([784,100], stddev=np.sqrt(2 / 784), seed=5566)
w2_initial = tf.truncated_normal([100,100], stddev=np.sqrt(2 / 100), seed=5566)
w3_initial = tf.truncated_normal([100,10], stddev=np.sqrt(2 / 100), seed=5566)

## ======================================
## (2) Try different activation functions
## ======================================
## Define leaky-relu with leak = 0.2 and __qualname = 'lrelu'
def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

act_fn_list = [tf.nn.relu, lrelu, tf.tanh, tf.nn.softplus]

w1_BN_dict, w1_grad_BN_dict, w2_BN_dict, w2_grad_BN_dict, accuracy_BN_dict, z2_BN_dict = {}, {}, {}, {}, {}, {}
w1_dict, w1_grad_dict, w2_dict, w2_grad_dict, accuracy_dict, z2_dict = {}, {}, {}, {}, {}, {}

for afn in act_fn_list:
    key = afn.__qualname__
    ## run model with BN:
    model = Model_DNN(n_steps = 40000, act_fn = afn, save_dir = save_dir, data=mnist)
    model.build_model(w1_initial,w2_initial,w3_initial)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.train(sess)
    w1_BN_dict[key] = model.w1_list
    w1_grad_BN_dict[key] = model.w1_grad_list        
    w2_BN_dict[key] = model.w2_list
    w2_grad_BN_dict[key] = model.w2_grad_list
    accuracy_BN_dict[key] = model.accuracy_list
    z2_BN_dict[key] = model.z2_list
    ## run model without BN:
    model = Model_DNN(n_steps = 40000, act_fn = afn, use_bn = False, save_dir = save_dir, data=mnist)
    model.build_model(w1_initial,w2_initial,w3_initial)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        model.train(sess)
    w1_dict[key] = model.w1_list
    w1_grad_dict[key] = model.w1_grad_list
    w2_dict[key] = model.w2_list
    w2_grad_dict[key] = model.w2_grad_list
    accuracy_dict[key] = model.accuracy_list
    z2_dict[key] = model.z2_list

## Plot accuracy
fig, ax = plt.subplots(figsize=(12,8))
color_list = ['b', 'g', 'r', 'k', 'c', 'm']

## (sort by accuracy in decreasing order)
final_accuracy_BN_dict = [accuracy_BN_dict[i][-1] for i in key_list]
decreasing_order = sorted(range(len(final_accuracy_BN_dict)),
                          key=lambda k: -final_accuracy_BN_dict[k])

## Plot accuracy with BN
for idx in decreasing_order:
    ax.plot(range(0,len(accuracy_BN_dict[key_list[idx]])*model.record_every_n_steps,model.record_every_n_steps),
            accuracy_BN_dict[key_list[idx]],
            color = color_list[idx], alpha = 0.4, linewidth = 2.5,
            label='activation function = %s, BN'%(key_list[idx]))

## Plot accuracy without BN
for idx in decreasing_order:
    ax.plot(range(0,len(accuracy_dict[key_list[idx]])*50,50),
            accuracy_dict[key_list[idx]],
            color = color_list[idx], alpha = 0.8, linewidth = 1.2,
            linestyle = '--',
            label='activation function = %s'%(key_list[idx]))

ax.set_xlabel('Training steps')
ax.set_ylabel('Accuracy')
# ax.set_xlim([20000,40000])
ax.set_ylim([0.92,1])
ax.set_title('Batch Normalization Accuracy')
ax.legend(loc=4)
plt.savefig(os.path.join(save_dir, 'accuracy_vs_act_fns.png'))
plt.show()
plt.close()


## Plot weight gradients of sigmoid,relu
mykey = ['sigmoid','relu']
for i in range(len(mykey)):
    w1_grad_list = w1_grad_dict[mykey[i]]
    w1_grad_BN_list = w1_grad_BN_dict[mykey[i]]
    w2_grad_list = w2_grad_dict[mykey[i]]
    w2_grad_BN_list = w2_grad_BN_dict[mykey[i]]
    plt.figure(1,figsize=(6,12))
    plt.subplot(211)
    plt.plot(np.mean(abs(w1_grad_list),axis=(1,2)),label="without BN")
    plt.plot(np.mean(abs(w1_grad_BN_list),axis=(1,2)),label="with BN")
    plt.xlabel("Training steps")
    plt.ylabel("Average magnitude of gradients")
    plt.title("layer 1 ("+mykey[i]+")")
    plt.legend(loc=1)
    plt.subplot(212)
    plt.plot(np.mean(abs(w2_grad_list),axis=(1,2)),label="without BN")
    plt.plot(np.mean(abs(w2_grad_BN_list),axis=(1,2)),label="with BN")
    plt.xlabel("Training steps")
    plt.ylabel("Average magnitude of gradients")
    plt.title("layer 2 ("+mykey[i]+")")
    plt.legend(loc=1)
    plt.savefig(os.path.join(save_dir, mykey[i]+"_w_grad_SGD.png"))
    plt.close()

## ------------------------------------------------------
## [NOTE] Save results before running the next experiment
## ------------------------------------------------------