import math
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris() #dataset IRIS
X = iris.data #numpy массив
Y = iris.target #numpy массив
N=49 #число значений K
Loo=np.empty(N) #массив, содержащий количество ошибок с некоторым K
def dist(a,b): #функция евклидова расстояния 
    r=0
    for i in range(len(a)):
        r+=(a[i]-b[i])**2
    return math.sqrt(r)    
def ker(r): #функция, которая вычисляет ядро Епаничникова
    if math.fabs(r)<=1: 
        return 3/4*(1-r**2)
    else:
        return 0
for k in range(1,3*N+1,3): #пробегаем по K
    h=0
    for i in range(len(Y)): #пробегаем по всем элементам датасета
        array1=np.empty((len(Y),2)) #создаём массив, содержащий расстояния от текущей точки до всех остальных
        for j in range(len(Y)): #пробегаем по точками датасета
            array1[j]=[j,dist(X[j,:],X[i,:])] #записываем расстояние в массив
        pd1=pd.DataFrame(array1) #массив переводим в DataFrame
        pd2=pd1.sort_values(by=1) #сортируем массив по значениям
        np1=np.array(pd2)
        xk1=np1[k+1,1] #элемент K+1
        np1=np1[1:k+1,:] #массив K элементов (расстояний)
        q=0
        r=0
        for j in range(3):
            w=0
            for m in range(k):#проходим по классам
                if j==Y[int(np1[m,0])]:
                    w+=ker(np1[m,1]/xk1)#находим вес для каждого класса на основе ядра
            if w>r:#выбираем наибольший вес
                q=j
                r=w
        if q!=Y[i]:#eсли выбранный класс отличается от истинного, увеличивается счетчик ошибок.
            h+=1
    Loo[int((k-1)/3)]=h #зписываем счётчик в массив
print(Loo)#выводим результат