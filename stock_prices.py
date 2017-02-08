#!/usr/bin/py

import numpy as np
#import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def printTransactions(m, k, d, name, owned, prices):
    transactions = []
    for data in range(k):
        model = make_pipeline(PolynomialFeatures(4), Ridge())
        X = np.array(range(1,6))[:,np.newaxis]
        y = prices[data][:5]
        model.fit(X, y)
        predicted_average_price = np.average(model.predict([[6],[7],[8],[9],[10]]))
        actual_current_price = prices[data][4]
        actual_average_price = np.average(y)

        #If current price is optimum, sell
        if actual_current_price > actual_average_price and actual_current_price > predicted_average_price:
            if owned[data] > 0:
                transactions.append((name[data],"SELL",owned[data]))
        #If current price is minimum, buy
        elif actual_current_price < actual_average_price and actual_current_price < predicted_average_price:
            transactions.append((name[data],"BUY",1))
        #If current price greater than actual average, sell
        elif actual_current_price > actual_average_price:
            if owned[data] > 0:
                transactions.append((name[data],"SELL",owned[data]))
        #If current price lesser than actual average, buy
        elif actual_current_price < actual_average_price:
            transactions.append((name[data],"BUY",1))
        else:
            if owned[data] > 0:
                transactions.append((name[data],"SELL",owned[data]))

    print len(transactions)
    for t in range(len(transactions)):
        print transactions[t][0],transactions[t][1],transactions[t][2]

if __name__ == '__main__':
    m, k, d = [float(i) for i in raw_input().strip().split()]
    k = int(k)
    d = int(d)
    names = []
    owned = []
    prices = []
    for data in range(k):
        temp = raw_input().strip().split()
        names.append(temp[0])
        owned.append(int(temp[1]))
        prices.append([float(i) for i in temp[2:7]])

    printTransactions(m, k, d, names, owned, prices)