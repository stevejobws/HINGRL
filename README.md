# HINGRL
## paper "HINGRL: Predicting Drug-disease Associations with Graph Representation Learning on Heterogeneous Information Networks"

### 'data' directory
Contain  B-Dataset and F-Dataset

### Install OpenNE
1. Refer to the [OpenNE](https://github.com/thunlp/OpenNE/tree/pytorch) configuration environment
2. To obtained drug and disease features by deepWalk, run
  - python -m openne --method deepWalk --input data/AllDrDiIs_train18416.txt --graph-format edgelist --output AllEmbedding_DeepWalk18416.txt --representation-size 64

### HINGRL.py
To predict drug-disease associations by HINGRL, run
  - python HINGRL.py -d 1 -f 10         

### Requirements
HINGRL is tested to work under Python 3.6.2  
The required dependencies for HINGRL  are Keras, PyTorch, TensorFlow, numpy, pandas, scipy, and scikit-learn.

### Contacts
If you have any questions or comments, please feel free to email BoWei Zhao (stevejobwes@gmail.com).
