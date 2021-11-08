# HINGRL
## Paper "HINGRL: Predicting Drug-disease Associations with Graph Representation Learning on Heterogeneous Information Networks"

### 'data' directory
Contain B-Dataset and F-Dataset

### 'Experimental Results' directory
1. Contain the prediction results of each baseline models is training on B-Dataset and F-Dataset
2. Please refer the code of LAGCN [here](https://github.com/storyandwine/LAGCN); please refer the code of DTINet [here](https://github.com/luoyunan/DTINet); please refer the code of deepDR [here](https://github.com/ChengF-Lab/deepDR).
3. Regarding the setting of parameters involved when running these algorithms, we either explicitly adopt the default setting recommended by their publications or conduct several trials with different settings to obtain the best performance. In which, we refer to [here](https://doi.org/10.1093/bib/bbab319) to use the ontology similarity of diseases as input features when LAGCN trained on the F-Dataset. For DTINet and deepDR, we only changed input dataset as B-Dataset and F-Dataset in the source code. 

### Install OpenNE
1. Refer to the [OpenNE](https://github.com/thunlp/OpenNE/tree/pytorch) configuration environment
2. To obtained drug and disease network representation by deepWalk, run
  - python -m openne --method deepWalk --input data/AllDrDiIs_train.txt --graph-format edgelist --output AllEmbedding_DeepWalk.txt --representation-size 64

### 'src' directory
1. Contain the source code of each classifier
2. To predict drug-disease associations by HINGRL, run
  - python HINGRL.py -d 1 -f 10 
  - -d is dataset selection, which B-Dataset is represented as 1 and F-Dataset is represented as 2. -f is fold number of cross-validation, and its default is 10.
3. How to tune the hyperparameters of each machine learning classifier, we give examples of scripts for each classifier, as shown below. Besides, we present the hyperparameters of all classifiers for HINGRL in Table 1.
4. Random Forest Classifier
  - python HINGRL.py -d 1 -f 10 -n 999
  - -d and -f as described above, -n is the number of tree of RandomForestClassifier, and its default is 999.
5. Gaussian Naïve Bayes
  - python Gaussian NB.py -d 1 -f 10
  - -d and -f as described above, its hyperparameters are default.
6. K Nearest Neighbor
  - python KNN.py -d 1 -f 10 -n 999
  - -d and -f as described above, -n is the number of neighbors, and its default is 9.
7. Logistic Regression
  - python LR.py -d 1 -f 10 -p l2
  - -d and -f as described above, -p is regularization parameter, and its default is l2.
8. Support Vector Machine
  - python SVM.py -d 1 -f 10 -k rbf -p l2
  - -d and -f as described above, -k is kernel function and its default is rbf, -p is regularization parameter and its default is l2.
9. Note that about more hyperparameters how to select it, please reference [sklearn](https://scikit-learn.org/).
### Table 1
| Classifiers | Hyperparameters |
| ----- | ----- |
| Gaussian NB | priors=None, var_smoothing=1e-9 |
| SVM | C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, probability=True, tol=1e-3, cache_size=200, max_iter=-1 |
| LR | penalty='l2', dual=False, tol=1e-4, C=1.0, fit_intercept=True, intercept_scaling=1, solver='lbfgs', max_iter=100 |
| KNN | n_neighbors=9, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski' |
| RF | n_estimators=999 |

### Options
See help for the other available options to use with *HINGRL*
  - python HINGRL.py --help

### Requirements
HINGRL is tested to work under Python 3.6.2  
The required dependencies for HINGRL are Keras, PyTorch, TensorFlow, numpy, pandas, scipy, and scikit-learn.

### Contacts
If you have any questions or comments, please feel free to email BoWei Zhao (stevejobwes@gmail.com).
