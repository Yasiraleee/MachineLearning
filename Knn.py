import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv("cancer.csv")
df.pop("Patient Id")
le=LabelEncoder()
df["Level"]=le.fit_transform(df["Level"])
print(df.info())
smk=SMOTETomek(random_state=42)
y=df.pop("Level")
df,y=smk.fit_resample(df,y)
print(y.value_counts())
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.2,random_state=1)
p=x_train.shape[0]
k=np.sqrt(p)
Knn=KNeighborsClassifier(n_neighbors=int(k))
kn1=Knn.fit(x_train,y_train)
pred=Knn.predict(x_test)
acc=accuracy_score(y_test,pred)
train_acc=Knn.score(x_train,y_train)
print(acc)
print(train_acc)
