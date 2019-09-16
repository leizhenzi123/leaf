import pandas as pd
from sklearn.preprocessing import  LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
def encode(train,test):
    le=LabelEncoder().fit(train.species)
    print(le)
    labels=le.transform(train.species)
    print(labels)
    classes=list(le.classes_)
    test_ids=test.id
    #print test_ids
    train=train.drop(['species','id'],axis=1)
    print (train)
    test=test.drop(['id'],axis=1)
    return train,labels,test,test_ids,classes
train, labels, test, test_ids, classes = encode(train, test)

