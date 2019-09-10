#import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC,NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import  LabelEncoder
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
def encode(train,test):
    le=LabelEncoder().fit(train.species)
    labels=le.transform(train.species)
    classes=list(le.classes_)
    test_ids=test.id
    train=train.drop(['species','id'],axis=1)
    test=test.drop(['id'],axis=1)
    return train,labels,test,test_ids,classes
train, labels, test, test_ids, classes = encode(train, test)
X_train, X_test = train.values[train.index], train.values[test.index]
y_train, y_test = labels[train.index], labels[test.index]
classifiers=[
    KNeighborsClassifier(3),
    SVC(kernel="rbf",probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    ]
log_cols=["Classifier","Accuracy","Log Loss"]
log=pd.DataFrame(columns=log_cols)
for clf in classifiers:
    clf.fit(X_train,y_train)
    name=clf.__class__.__name__
    print("="*30)
    print(name)
    print('*****Results****')
    train_predictions=clf.predict(X_test)
    acc=accuracy_score(y_test,train_predictions)
    print("Accuracy:{:.4%}".format(acc))
    train_predictions=clf.predict_proba(X_test)
    ll=log_loss(y_test,train_predictions)
    print ("Log Loss:{}".format(ll))
    log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_cols)
    log = log.append(log_entry)
print("="*30)
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")
plt.xlabel('Accuracy %')
plt.title('Classifier performance comparison')
plt.show()
sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")
plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.show()
favorite_clf = LinearDiscriminantAnalysis()
favorite_clf.fit(X_train, y_train)
test_predictions = favorite_clf.predict_proba(test)
submission = pd.DataFrame(test_predictions, columns=classes)
submission.insert(0, 'id', test_ids)
submission.reset_index()
submission.tail()
