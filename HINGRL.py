# coding: utf-8
import pandas as pd

DrDiNum18416 = pd.read_csv('./data/B-Dataset/HINGE_DrugDieaseNum18416.csv',header=None)
DrPrNum3243 = pd.read_csv('./data/B-Dataset/HINGE_DrugProteinNum18416.csv',header=None)
DiPrNum71840 = pd.read_csv('./data/B-Dataset/HINGE_DiseaseProteinNum18416.csv',header=None)

import math
import random
def partition(ls, size):
    return [ls[i:i+size] for i in range(0, len(ls), size)]
RandomList = random.sample(range(0, len(DrDiNum18416)), len(DrDiNum18416))
print('len(RandomList)', len(RandomList))
NewRandomList = partition(RandomList, math.ceil(len(RandomList) / 10))
print('len(NewRandomList[0])', len(NewRandomList[0]))
#NaN = pd.isnull(NewRandomList).any(0).nonzero()[0]
NewRandomList = pd.DataFrame(NewRandomList)
NewRandomList = NewRandomList.fillna(int(0))
NewRandomList = NewRandomList.astype(int)
NewRandomList.to_csv('./data/HINGE_NewRandomList18416.csv', header=None,index=False)
del NewRandomList, RandomList

import numpy as np
Nindex = pd.read_csv('./data/HINGE_NewRandomList18416.csv',header=None)
for i in range(len(Nindex)):
    kk = []
    for j in range(10):
        if j !=i:
            kk.append(j)
    index = np.hstack([np.array(Nindex)[kk[0]],np.array(Nindex)[kk[1]],np.array(Nindex)[kk[2]],np.array(Nindex)[kk[3]],np.array(Nindex)[kk[4]],
                       np.array(Nindex)[kk[5]],np.array(Nindex)[kk[6]],np.array(Nindex)[kk[7]],np.array(Nindex)[kk[8]]])
    DTIs_train= pd.DataFrame(np.array(DrDiNum18416)[index])
    DTIs_train.to_csv('./data/DrDiIs_train'+str(i)+'.csv', header=None,index=False)
    DTIs_train = DTIs_train.append(DrPrNum3243.append(DiPrNum71840))
    DTIs_train = DTIs_train.sample(frac=1.0)
    DTIs_train.to_csv('./data/DrDiIs_train'+str(i)+'.txt', sep='\t' ,header=None,index=False)
    DTIs_test=pd.DataFrame(np.array(DrDiNum18416)[np.array(Nindex)[i]])
    DTIs_test.to_csv('./data/DrDiIs_test'+str(i)+'.csv', header=None,index=False)
    print(i)
del Nindex, index, DTIs_train, DTIs_test

import numpy as np
DTIs_train = DrDiNum18416.append(DrPrNum3243.append(DiPrNum71840))
DTIs_train = DTIs_train.sample(frac=1.0)
DTIs_train.to_csv('./data/AllDrDiIs_train18416.txt', sep='\t' ,header=None,index=False)


def NegativeGenerate(LncDisease, AllRNA,AllDisease):
    import random
    NegativeSample = []
    counterN = 0
    while counterN < len(LncDisease): 
        counterR = random.randint(0, len(AllRNA) - 1)
        counterD = random.randint(0, len(AllDisease) - 1)
        DiseaseAndRnaPair = []
        DiseaseAndRnaPair.append(AllRNA[counterR])
        DiseaseAndRnaPair.append(AllDisease[counterD])
        flag1 = 0
        counter = 0
        while counter < len(LncDisease):
            if DiseaseAndRnaPair == LncDisease[counter]:
                flag1 = 1
                break
            counter = counter + 1
        if flag1 == 1:
            continue
        flag2 = 0
        counter1 = 0
        while counter1 < len(NegativeSample): 
            if DiseaseAndRnaPair == NegativeSample[counter1]:
                flag2 = 1
                break
            counter1 = counter1 + 1
        if flag2 == 1:
            continue
        if (flag1 == 0 & flag2 == 0):
            NamePair = [] 
            NamePair.append(AllRNA[counterR])
            NamePair.append(AllDisease[counterD])
            NegativeSample.append(NamePair)
            counterN = counterN + 1
    return NegativeSample
Dr = pd.read_csv('./data/B-Dataset/HINGE_drugName18416.csv',header=0,names=['id','name'])
Pr = pd.read_csv('./data/B-Dataset/HINGE_diseaseName18416.csv',header=0,names=['id','name'])
NegativeSample = NegativeGenerate(DrDiNum18416.values.tolist(),Dr['id'].values.tolist(),Pr['id'].values.tolist())
NegativeSample = pd.DataFrame(NegativeSample)
NegativeSample.to_csv('./data/HINGE_NegativeSample18416.csv', header=None,index=False)

import scipy.sparse as sp
import pandas as pd
import numpy as np
creat_var = locals()
creat_var = locals()
Negative = pd.read_csv('./data/HINGE_NegativeSample18416.csv',header=None)
Nindex = pd.read_csv('./data/HINGE_NewRandomList18416.csv',header=None)
Attribute = pd.read_csv('./data/B-Dataset/HINGE_AllNodeAttribute18416.csv',header = None, index_col=0)
Attribute = Attribute.iloc[:,1:]
Embedding = pd.read_csv('./data/AllEmbedding_DeepWalk18416.txt', sep=' ',header=None,skiprows=1)
Embedding = Embedding.sort_values(0,ascending=True)
Embedding.set_index([0], inplace=True)
Negative[2] = Negative.apply(lambda x: 0 if x[0] < 0 else 0, axis=1)
for i in range(10):
    train_data = pd.read_csv('./data/DrDiIs_train'+str(i)+'.csv',header=None)
    train_data[2] = train_data.apply(lambda x: 1 if x[0] < 0 else 1, axis=1)
    kk = []
    for j in range(10):
        if j !=i:
            kk.append(j)
    index = np.hstack([np.array(Nindex)[kk[0]],np.array(Nindex)[kk[1]],np.array(Nindex)[kk[2]],np.array(Nindex)[kk[3]],np.array(Nindex)[kk[4]],
                       np.array(Nindex)[kk[5]],np.array(Nindex)[kk[6]],np.array(Nindex)[kk[7]],np.array(Nindex)[kk[8]]])
    result = train_data.append(pd.DataFrame(np.array(Negative)[index]))
    labels_train = result[2]
    data_train_feature = pd.concat([pd.concat([Attribute.loc[result[0].values.tolist()],Embedding.loc[result[0].values.tolist()]],axis=1).reset_index(drop=True),
           pd.concat([Attribute.loc[result[1].values.tolist()],Embedding.loc[result[1].values.tolist()]],axis=1).reset_index(drop=True)],axis=1)
    
    creat_var['data_train'+str(i)] = data_train_feature.values.tolist()
    creat_var['labels_train'+str(i)] = labels_train
    print(len(labels_train))
    del labels_train, result, data_train_feature
    test_data = pd.read_csv('./data/DrDiIs_test'+str(i)+'.csv',header=None)
    test_data[2] = test_data.apply(lambda x: 1 if x[0] < 0 else 1, axis=1)
    result = test_data.append(pd.DataFrame(np.array(Negative)[np.array(Nindex)[i]]))    
    labels_test = result[2]
    data_test_feature = pd.concat([pd.concat([Attribute.loc[result[0].values.tolist()],Embedding.loc[result[0].values.tolist()]],axis=1).reset_index(drop=True),
           pd.concat([Attribute.loc[result[1].values.tolist()],Embedding.loc[result[1].values.tolist()]],axis=1).reset_index(drop=True)],axis=1)
    
    creat_var['data_test'+str(i)] = data_test_feature.values.tolist()
    creat_var['labels_test'+str(i)] = labels_test
    print(len(labels_test))
    del train_data, test_data, labels_test, result, data_test_feature
    print(i)


data_train = [data_train0,data_train1,data_train2,data_train3,data_train4,data_train5,data_train6,data_train7,data_train8,data_train9]
data_test = [data_test0,data_test1,data_test2,data_test3,data_test4,data_test5,data_test6,data_test7,data_test8,data_test9]
labels_train = [labels_train0,labels_train1,labels_train2,labels_train3,labels_train4,labels_train5,labels_train6,labels_train7,labels_train8,labels_train9]
labels_test = [labels_test0,labels_test1,labels_test2,labels_test3,labels_test4,labels_test5,labels_test6,labels_test7,labels_test8,labels_test9]


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from scipy import interp
import time

now = time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime(time.time()))


print("10-CV")
tprs=[]
aucs=[]
mean_fpr=np.linspace(0,1,1000)

colorlist = ['red','firebrick', 'gold','limegreen','royalblue', 'purple', 'green','magenta', 'blue', 'black']

AllResult = []


for i in range(10):

    X_train,X_test = data_train[i],data_test[i]
    Y_train,Y_test = np.array(labels_train[i]),np.array(labels_test[i])
    best_RandomF = RandomForestClassifier(n_estimators=999,n_jobs=-1)
    print('Strat training')
    best_RandomF.fit(np.array(X_train), np.array(Y_train))

    y_score0 = best_RandomF.predict(np.array(X_test))
    y_score_RandomF = best_RandomF.predict_proba(np.array(X_test))
    fpr,tpr,thresholds=roc_curve(Y_test,y_score_RandomF[:,1])
    tprs.append(interp(mean_fpr,fpr,tpr))
    tprs[-1][0]=0.0
    #auc
    roc_auc=auc(fpr,tpr)
    aucs.append(roc_auc)
    print('ROC fold %d(AUC=%0.4f)'% (i,roc_auc)) 
    
mean_tpr=np.mean(tprs,axis=0)
mean_tpr[-1]=1.0
mean_auc=auc(mean_fpr,mean_tpr)
print('Mean ROC (AUC=%0.4f)'% (mean_auc)) 
