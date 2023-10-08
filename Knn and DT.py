import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt

#EDA
df=pd.read_csv("health.csv")
x=df.isnull().sum().sum()
print(x)
df=df.dropna()
print(df.isnull().sum().sum())

#balance check
x=df["stroke"].value_counts()
print(x)
le=LabelEncoder()
df["stroke"]=le.fit_transform(df["stroke"])
df["smoking_status"]=le.fit_transform(df["smoking_status"])
df["Residence_type"]=le.fit_transform(df["Residence_type"])
df["work_type"]=le.fit_transform(df["work_type"])
df["ever_married"]=le.fit_transform(df["ever_married"])
df["gender"]=le.fit_transform(df["gender"])
smk=SMOTETomek(random_state=42)
y=df.pop("stroke")
df,y=smk.fit_resample(df,y)
x=y.value_counts()
print(x)
print(df.info())


#decision tree
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.2)
dt=DecisionTreeClassifier(criterion="entropy")
d=dt.fit(x_train,y_train)
pred=dt.predict(x_test)
acc=accuracy_score(y_test,pred)
print(acc)

#Knn
p=x_train.shape[0]
k=np.sqrt(p)
Knn=KNeighborsClassifier(n_neighbors=int(k))
kn=Knn.fit(x_train,y_train)
pred=Knn.predict(x_test)
acc=accuracy_score(y_test,pred)
print(acc)
