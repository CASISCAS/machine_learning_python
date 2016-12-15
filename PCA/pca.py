# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 19:42:23 2016

@author:june

code of PCA Algrithom
"""
import numpy as np

from util.tools import scale


class PCA():

    def __init__(self, percent=0.99, rowvar=0):
        self.percent = percent # 根据要求的方差百分比，求出所需要的特征值的个数n
        self.rowvar = rowvar #若rowvar非0，一列代表一个样本，为0，一行代表一个样本

    def percent2n(self, eigVals):
        sortArray = np.sort(eigVals)  # 升序
        sortArray = sortArray[-1::-1]  # 逆转，即降序

        arraySum_needed = sum(sortArray) * self.percent
        tmp = 0
        num = 0
        for i in sortArray:
            tmp += i
            num += 1
            if tmp >= arraySum_needed:
                return num

    def pca(self, dataMat):

        newData= scale(dataMat)
        covMat = np.cov(newData, rowvar=self.rowvar)  # 求协方差矩阵,return ndarray；

        eigVals, eigVects = np.linalg.eig(np.mat(covMat))  # 求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
        n = self.percent2n(eigVals)  # 要达到percent的方差百分比，需要前n个特征向量
        eigValIndice = np.argsort(eigVals)  # 对特征值从小到大排序
        n_eigValIndice = eigValIndice[-1:-(n + 1):-1]  # 最大的n个特征值的下标
        n_eigVect = eigVects[:, n_eigValIndice]  # 最大的n个特征值对应的特征向量
        lowDDataMat = newData * n_eigVect  # 低维特征空间的数据
        reconMat = (lowDDataMat * n_eigVect.T) + np.mean(dataMat)  # 重构数据

        return lowDDataMat, reconMat


def demo():

    data = np.loadtxt("testSet.txt")

    pca = PCA(rowvar=0)
    lowdata, reconmat = pca.pca(data)

    print(lowdata.size)

def sklearn_demo():

    from sklearn.decomposition import PCA
    data = np.loadtxt("testSet.txt")
    pca = PCA(2)
    pca.fit(data)
    print(pca.explained_variance_ratio_)




if __name__ == "__main__":
    #demo()
    sklearn_demo()
