# Tensorflow-NMT-Tutorial-PTB

### Overview ###
- This repository contains preprocessing and postprocessing code for Penn Treebank data to prepare the data for use with the Tensorflow Neural Machine Translation tutorial (See https://github.com/tensorflow/nmt).
- The preprocessing and postprocessing code have been restored after a computer failure.
- The repository contains EVALB code for evaluation. *postprocess.py* will run this automatically.

### Usage ###
- The repository does *NOT* contain *nmt*. In order to run the code, you will need to clone this repository and then clone https://github.com/tensorflow/nmt into the uppermost level of the cloned Tensorflow-NMT-Tutorial-PTB repository.
- Run *preprocess.py* first, followed by *postprocess.py*. Because of incompatibility issues, please use Python 3.6 and TensorFlow 1.12.0 (do not use TensorFlow 2.0).

### Notes ###
For more information, please see the enclosed write-up.
