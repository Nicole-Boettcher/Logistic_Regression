# Logistic Regression for Binary Classification

import numpy as np
from numpy import random
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

def fetch_data():
    from sklearn.datasets import load_breast_cancer
    breast_cancer = load_breast_cancer()
    print(breast_cancer.DESCR)
    X, t = load_breast_cancer(return_X_y=True)
    print(X.shape)
    print(t.shape)

def main():
    print("Assignment 2")

    # fetch data 
    fetch_data()
    
main()