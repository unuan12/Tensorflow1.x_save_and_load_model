## Tensorflow1.x save and load model

This repository is built to learn how tensorflow 1.x load models.

#### Environment

+ python 3.6
+ tensorflow-gpu 1.13

#### Usage

run `mnist_train.py` to get trained models.

To use trained model in another program, you must re-create the associated graph structure (e.g. by running code to build it again, or use `.meta` .

`test_with_network.py` tests by rebuilding the network.

`test_with_meta.py` tests by loading `.meta`.

And there is a wrong use in `incorrect_usage.py`，which both rebuilding the network and loading `.meta`.

you can use `Tensorboard` to view network structure，such as 

`tensorboard --logdir=./log/test_with_meta`

#### Result

Select the top 5000 MNIST datasets for testing.

| loading mode      | accuracy |
| ----------------- | -------- |
| test_with_network | 0.9784   |
| test_with_meta    | 0.9784   |
| incorrect_usage   | 0.0982   |

