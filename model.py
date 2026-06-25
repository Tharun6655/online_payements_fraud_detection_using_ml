import numpy as np 
import pandas as pd 
import matplotlib.pyplot as  plt 
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_auc_score as ras 
from xgboost import XGBClassifier
from sklearn.metrics import ConfusionMatrixDisplay





data=pd.read_csv('./data/new_file.csv')
print(data.head())

print(data.describe()) 

print(data.info())

obj=(data.dtypes=='str') 
obj_cols=list(obj[obj].index)
print('Catgorial data:',len(obj_cols))

Int=(data.dtypes=='int')
Int_cols=list(Int[Int].index)
print('Int data types',len(Int_cols))

fl=(data.dtypes=='float')
fl_cols=list(fl[fl].index)
print('Float data types',len(fl_cols))

#count of the types of payment 
sns.countplot(x='type', data=data) 
plt.title('Count of types of payment')
plt.show() 

sns.barplot(x='type', y='amount', data=data)
plt.title('Count and types of payment')
plt.show()

f=data['isFraud'].value_counts()
print(f)

#distribution of the step data 
plt.figure(figsize=(16,6))
sns.displot(data['step'],bins=50)
plt.show()


plt.figure(figsize=(12, 6))
sns.heatmap(data.apply(lambda x: pd.factorize(x)[0]).corr(),
			cmap='BrBG',
			fmt='.2f',
			linewidths=2,
			annot=True)
plt.title('correlation matrix amoung the features')
plt.show()

type_new = pd.get_dummies(data['type'], drop_first=True)
data_new = pd.concat([data, type_new], axis=1)
data_new.head()

X=data_new.drop(['isFraud','type','nameOrig','nameDest'],axis=1)
y=data_new['isFraud']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)


print(X.shape)
print(y.shape) 

models=[LogisticRegression(), XGBClassifier(),RandomForestClassifier(n_estimators=7,criterion='entropy',random_state=7)]

for i in range(len(models)):
    models[i].fit(X_train,y_train)
    print(f'{models[i]}')

    train_pred=models[i].predict_proba(X_train)[:,1]
    print('Training Accuracy',ras(y_train,train_pred))

    y_preds=models[i].predict_proba(X_test)[:,1]
    print('Validation result',ras(y_test,y_preds))
    print()


cm=ConfusionMatrixDisplay.from_estimator(models[1],X_test,y_test)
cm.plot(cmap='Blues')
plt.show()

