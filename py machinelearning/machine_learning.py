#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 19:55:39 2023

@author: batuhanalpkurban
"""
# ders 6 : kütüphanelerin yüklenmesi
##kütüphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('eksikveriler.csv')

print(veriler)

boy = veriler[['boy']]
print(boy)

boykilo = veriler[['boy','kilo']]
print(boykilo)

x = 10

class insan:
    boy = 190
    def kosmak(self,b):
        return b+10

alp = insan()
print(alp.boy)
print(alp.kosmak(100))

##eksik veriler
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

Yas = veriler.iloc[:,1:4].values
print(Yas)

imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])
print(Yas)

ulke = veriler.iloc[:,0:1].values
print(ulke)

from sklearn import preprocessing

le= preprocessing.LabelEncoder()

ulke [:,0] = le.fit_transform(veriler.iloc[:,0])

print (ulke)

ohe = preprocessing. OneHotEncoder() 

ulke = ohe.fit_transform(ulke).toarray()

print(ulke)

print(list(range(22)))

sonuc = pd.DataFrame(data=ulke, index = range (22), columns = ['fr', 'tr', 'us'])

print(sonuc)

sonuc2 = pd.DataFrame(data=Yas, index = range (22), columns = ['boy', 'kilo', 'yas'])

print (sonuc2)

cinsiyet = veriler.iloc[:, -1].values

print(cinsiyet)

sonuc3= pd.DataFrame(data = cinsiyet, index = range(22), columns = ['cinsiyet'])
print(sonuc3)

s=pd.concat([sonuc,sonuc2], axis=1)
print(s)

s2 = pd.concat([s,sonuc3], axis=1)
print(s2)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3,test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler

sc=StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
