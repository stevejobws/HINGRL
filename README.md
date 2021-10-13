# HINGRL
## paper "HINGRL: Predicting Drug-disease Associations with Graph Representation Learning on Heterogeneous Information Networks"

### 'data' directory
Contain  B-Dataset and F-Dataset

### install OpenNE
1. Refer to the [OpenNE](https://github.com/thunlp/OpenNE/tree/pytorch) configuration environment
2. To obtained drug and disease features by deepWalk, run
  - python -m openne --method deepWalk --input data/AllDrDiIs_train.txt --graph-format edgelist --output AllEmbedding_DeepWalk.txt --representation-size 64

### HINGRL.py
To predict drug-disease associations by HINGRL, run
  - pretraining with features: python HINGRL.py.py

### Requirements
HINGRL is tested to work under Python 3.6.2  
The required dependencies for HINGRL  are Keras, PyTorch, TensorFlow, numpy, pandas, scipy, and scikit-learn.
