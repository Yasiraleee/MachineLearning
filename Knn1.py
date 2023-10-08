import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder
from imblearn.combine import SMOTETomek
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("DataTraining.csv")
df2=pd.read_csv("DataTest.csv")
import numpy as np
x_train=df[["Humidity", "Light", "HumidityRatio"]]
x_test=df2[["Humidity", "Light", "HumidityRatio"]]
y_train=df["Occupancy"]
y_test=df2["Occupancy"]
p=x_train.shape[0]
k=np.sqrt(p)
Knn=KNeighborsClassifier(n_neighbors=int(k))
kn1=Knn.fit(x_train,y_train)
pred=Knn.predict(x_test)
acc=accuracy_score(y_test,pred)
print(acc)