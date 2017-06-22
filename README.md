
# Why Batch Normalization Works So Well?
Team: 我才是真的 Baseline

Team Members: f03942038 鍾佳豪,r05942102 王冠驊,d05921018 林家慶,d05921027 張鈞閔

## Quick Start
1. The effect of different activation functions on batch normalization
```
python exp_actfn.py
```

2. The effect of different optimizer on batch normalization
```
python exp_optimizer.py
```

3. The effect of different batch size on batch normalization
```
python exp_batch_size.py
```

4. The effect of different training/testing mismatch on batch normalization
```
python exp_mismatch.py
```

5. Regularization effect
```
Note that the results are observable from the experiment of different activation functions (1)
```

6. How batch normalization influences singular values of layer Jacobian?
```
python exp_Jacobian.py
```

Note that the resulting plots will be stored in save_dir.

### previous version of our implementation
```
python model_DNN_example.py
```
All plots will be saved in "save_DNN/"
