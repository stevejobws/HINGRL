# HINGRL
## paper "HINGRL: Predicting Drug-disease Associations with Graph Representation Learning on Heterogeneous Information Networks"

### 'data' directory
Contain B-Dataset and F-Dataset

### 'compareModel' directory
Contain the prediction results of each baseline models is training on B-Dataset and F-Dataset

### Install OpenNE
1. Refer to the [OpenNE](https://github.com/thunlp/OpenNE/tree/pytorch) configuration environment
2. To obtained drug and disease network representation by deepWalk, run
  - python -m openne --method deepWalk --input data/AllDrDiIs_train.txt --graph-format edgelist --output AllEmbedding_DeepWalk.txt --representation-size 64

### HINGRL.py
To predict drug-disease associations by HINGRL, run
  - python HINGRL.py -d 1 -f 10 
  - -d is dataset selection, which B-Dataset is represented as 1 and F-Dataset is represented as 2. -f is fold number of cross-validation. default is 10.   

### Options
See help for the other available options to use with *HINGRL*
  - python HINGRL.py --help

### Requirements
HINGRL is tested to work under Python 3.6.2  
The required dependencies for HINGRL are Keras, PyTorch, TensorFlow, numpy, pandas, scipy, and scikit-learn.

### Contacts
If you have any questions or comments, please feel free to email BoWei Zhao (stevejobwes@gmail.com).
