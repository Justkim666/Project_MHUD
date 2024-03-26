import pandas as pd

# a) doc du lieu ruou vang DO
dt = pd.read_csv("telecom_churn_after_format.csv")
print("So luong phan tu:",len(dt)) # 

import numpy as np
# cac gia tri khac nhau cua nhan
print("So luong nhan:", len(np.unique(dt.Churn)) )
print(np.unique(dt.Churn))
# so luong nhan
print("So luong phan tu tung nhan:")
print(dt.Churn.value_counts())

print("Du lieu 5 hang dau tien")
print(dt.head(5))


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB


#Phan chia tap du lieu
x_train, x_test, y_train, y_test = train_test_split(dt.iloc[:,1:11],dt.Churn,test_size=1/3.0,shuffle=1,random_state=25)

# print("Thuoc tinh dung de huan luyen la: ")
# print(x_train)
# print("Nhan dung de huan luyen la: ")
# print(y_train)
# print("Thuoc tinh dung de test la: ")
# print(x_test)
# print("Nhan dung de test la: ")
# print(y_test)

# Xay dung mo hinh KNN voi 15 lang gieng
moHinhKNN_Holdout = KNeighborsClassifier(n_neighbors=15)
moHinhKNN_Holdout.fit(x_train,y_train)


# do chinh xac tong the KNN
y_pred_knn = moHinhKNN_Holdout.predict(x_test)
knn_score = accuracy_score(y_pred_knn,y_test)*100
print("Accuracy score for KNN is: %.4f" %knn_score)
print("Confusion matrix")
print(confusion_matrix(y_test, y_pred_knn, labels=[0,1]))

# print("========= Du Doan Phan Tu Moi Den ===========")
# moHinhKNN_Holdout.fit(x_train.values,y_train.values)

# duDoan = moHinhKNN_Holdout.predict([[0.58,0,1,0.56,0.0,0.74,0.51,0.81,0.61,0.56]])
# print("Nhan cua phan tu moi den la:", duDoan)


for i in range(11,31,2):

 x_train, x_test, y_train, y_test = train_test_split(dt.iloc[:,1:11],dt.Churn,test_size=1/3.0,shuffle=1,random_state=25+i)
 # Xay dung mo hinh KNN voi 15 lang gieng
 moHinhKNN_Holdout = KNeighborsClassifier(n_neighbors=15)
 moHinhKNN_Holdout.fit(x_train,y_train)

 # do chinh xac tong the KNN
 y_pred_knn = moHinhKNN_Holdout.predict(x_test)
 knn_score = accuracy_score(y_pred_knn,y_test)*100
 print("n_neighbors =",i)
 print("Accuracy score for KNN is: %.4f" %knn_score)
 print("Confusion matrix")
 print(confusion_matrix(y_test, y_pred_knn, labels=[0,1]))
 # Xay dung mo hinh bayes theo phan phoi Gaussian
 moHinhBayes_Holdout = GaussianNB()
 moHinhBayes_Holdout.fit(x_train,y_train)
 #bayes
 # do chinh xac tong the Bayes
 y_pred_bayes = moHinhBayes_Holdout.predict(x_test)
 gaussianNG_score = accuracy_score(y_pred_bayes,y_test)*100
 print("Accuracy score for Bayes is: %.4f" %gaussianNG_score)
 print(confusion_matrix(y_test, y_pred_bayes,labels=[0,1]))

 ## So sanh ket qua

 if knn_score > gaussianNG_score : print("Do chinh xac KNN > GaussianNB {:.4f} > {:.4f}".format(knn_score, gaussianNG_score))
 if knn_score < gaussianNG_score : print("Do chinh xac KNN < GaussianNB {:.4f} < {:.4f}".format(knn_score, gaussianNG_score))
 if knn_score == gaussianNG_score : print("Do chinh xac KNN = GaussianNB {:.4f} = {:.4f}".format(knn_score, gaussianNG_score))