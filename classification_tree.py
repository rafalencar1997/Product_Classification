from __future__ import print_function

import numpy as np
from numpy import random
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

class Tree:
    def __init__(self, cargo, branch=None, model=None):
        self.cargo  = cargo
        self.branch = branch
        self.model = model

    def getCargo(self):
        return str(self.cargo)
    
    def setCargo(self, value):
        self.cargo = value
    
    def getModel(self):
        return self.model
    
    def setModel(self, model):
        self.model = model
    
    def getNode(self, index):
        if self.branch != None:
            return self.branch[index]
        else:
            return None
        
    def getBranchStr(self):
        br_list = []
        if self.branch:
            for br in self.branch:
                br_list.append(br.getCargo())
            return br_list
        else:
            return None
    
    def getBranchLen(self):
        if self.branch:
            return len(self.branch)
        else: 
            return 0
    
    def findNode(self, value):
        try:
            for i, br in enumerate(self.branch):
                if br.getCargo() == value:
                    return i
            return -1
        except TypeError:
            return -1
    
    def addNode(self, node):
        if node:
            if not self.branch:
                self.branch = []
            self.branch.append(node)
            
    def setNode(self, index, node):
        if node:
            if self.branch:
                self.branch[index] = node
        else:
            return None
        
    def populateTree(self, split, index):
        if index == (len(split)):
            item = split[index-1]
            return Tree(item)
        else:
            item = split[index]
        brIndex = self.findNode(item) 
        if brIndex < 0:
            newNode = Tree(item).populateTree(split, index+1)
            self.addNode(newNode)
        else:
            if(index < len(split)-1):
                newNode = self.getNode(brIndex).populateTree(split, index+1)
                self.setNode(brIndex, newNode)
        return self

    def printTree(self, h):
        print('|  '*h, '>', self.getCargo(), str(self.getBranchLen()))
        if not self.branch:
            return
        for branch in self.branch:
            branch.printTree(h+1)

    def modelTree(self, dataX, dataY, h):
        X = dataX
        Y = dataY
        Xshape = np.shape(X)
        Yshape = np.shape(Y)
        print('|  '*h, '>', self.getCargo(), 
              'X:', Xshape, 'Y:', Yshape, 'N', str(self.getBranchLen()))
        if not self.branch:
            return self
        for i, branch in enumerate(self.branch):
            if Yshape[0] > 0:
                newX, newY = separate_data(X, Y, branch.getCargo())
            else:
                newX, newY = X, Y
            newNode = branch.modelTree(newX, newY, h+1)
            self.setNode(i, newNode)
        if Yshape[0] > 0:
            Y = separate_label(Y) 
            self.setModel(MultinomialNB(alpha=0.005).fit(X, Y))
        return self
    
    def sampleTree(self, dataX, dataY, h):
        X = dataX
        Y = dataY
        Xshape = np.shape(X)
        Yshape = np.shape(Y)
        print('|  '*h, '>', self.getCargo(), 
              'Samples:', Xshape[0], 'Nodes', str(self.getBranchLen()))
        if not self.branch:
            return self
        for i, branch in enumerate(self.branch):
            if Yshape[0] > 0:
                newX, newY = separate_data(X, Y, branch.getCargo())
            else:
                newX, newY = X, Y
            newNode = branch.sampleTree(newX, newY, h+1)
            self.setNode(i, newNode)
        return self
    
    def scoreTree(self, dataX, dataY, h):
        X = dataX
        Y = dataY
        Xshape = np.shape(X)
        Yshape = np.shape(Y)
        model = self.getModel()
        score = None
        if model and Yshape[0] > 0:
            labelY = separate_label(Y) 
            score = 100*round(model.score(X, labelY), 4)
        print('|  '*h, '>', self.getCargo(), 
              'X:', Xshape, 'Y:', Yshape, 'S', str(score))
        if not self.branch:
            return
        for i, branch in enumerate(self.branch):
            if Yshape[0] > 0:
                newX, newY = separate_data(X, Y, branch.getCargo())
            else:
                newX, newY = X, Y
            branch.scoreTree(newX, newY, h+1)

    def fullScoreTree(self, dataX, dataY):
        count = 0.0
        acc = 0.0
        h_acc = np.zeros(5)
        h_count = np.zeros(5)
        for i, X in enumerate(dataX.values):
            node = self
            Y = dataY.values
            split = Y[i].split(' > ')
            for j, label in enumerate(split):
                prediction = node.getModel().predict(X.reshape(1,X.shape[0]))
                h_count[j] += 1
                if prediction == label:
                    h_acc[j] += 1
                    if j == (len(split)-1):
                        acc += 1.0
                    node = node.getNode(node.findNode(label))
                else:
                    break
            count += 1.0 
        return count, acc, h_count, h_acc 

def separate_data(dataX, dataY, category):
    A = dataY.str.split(' > ', n = 1, expand = True)
    B = A[:][0] == category
    DX = dataX.loc[B.values]
    DY = dataY.loc[B.values]
    A  = A.loc[B.values]
    lenq = np.shape(A)[-1]
    DY = (A[:][lenq-1]).replace(np.nan, '', regex=True)
    return DX, DY

def separate_label(dataY):
    A = dataY.str.split(' > ', n = 1, expand = True)
    DY = (A[:][0]).replace(np.nan, '', regex=True)
    return DY