import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df_train=pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")
X_test=df_test

X_train=df_train.drop('Survived',axis=1)

X_train=X_train.drop('Name',axis=1)
X_test=X_test.drop('Name',axis=1)

one_hot_Xtrain_Sex=pd.get_dummies(X_train['Sex'])
one_hot_Xtest_Sex=pd.get_dummies(X_test['Sex'])

X_train=pd.concat([X_train,one_hot_Xtrain_Sex],axis=1)
X_test=pd.concat([X_test,one_hot_Xtest_Sex],axis=1)

X_train=X_train.drop('Sex',axis=1)
X_test=X_test.drop('Sex',axis=1)

X_train=X_train.drop('Ticket',axis=1)
X_test=X_test.drop('Ticket',axis=1)

X_train=X_train.drop('Cabin',axis=1)
X_test=X_test.drop('Cabin',axis=1)

one_hot_Xtrain_Embarked=pd.get_dummies(X_train['Embarked'])
one_hot_Xtest_Embarked=pd.get_dummies(X_test['Embarked'])

X_train=pd.concat([X_train,one_hot_Xtrain_Embarked],axis=1)
X_test=pd.concat([X_test,one_hot_Xtest_Embarked],axis=1)

X_train=X_train.drop('Embarked',axis=1)
X_test=X_test.drop('Embarked',axis=1)

for column in X_train.columns:
    if X_train[column].dtype == 'float64':
        X_train[column]=X_train[column].fillna(X_train[column].median())
    elif X_train[column].dtype == 'object':
        X_train[column]=X_train[column].fillna(X_train[column].mode()[0])
    else:
        X_train[column]=X_train[column].fillna(False)

for column in X_test.columns:
    if X_test[column].dtype == 'float64':
        X_test[column]=X_test[column].fillna(X_test[column].median())
    elif X_test[column].dtype == 'object':
        X_test[column]=X_test[column].fillna(X_test[column].mode()[0])
    else:
        X_test[column]=X_test[column].fillna(False)

y_train=df_train['Survived']
print("X_train:",X_train)

#Feature Scaling
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.fit_transform(X_test)

model=LogisticRegression(max_iter=1000)
model.fit(X_train_scaled,y_train)
y_pred=model.predict(X_test_scaled)
print(y_pred)

print('df_test_PassengerId',df_test['PassengerId'])
submitted_csv=pd.DataFrame({
    'PassengerId':df_test['PassengerId'],
    'Survived':y_pred
})
print(submitted_csv)
submitted_csv.to_csv('titanic.csv',index=False)
