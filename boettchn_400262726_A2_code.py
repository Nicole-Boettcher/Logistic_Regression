# Logistic Regression for Binary Classification

import numpy as np
from numpy import random
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

sc = StandardScaler()

random_seed = 2726

def fetch_data():
    breast_cancer = load_breast_cancer()
    print(breast_cancer.DESCR)
    X, t = load_breast_cancer(return_X_y=True)
    #print("Shape of X: ", X.shape)
    #print("Shape of t: ", t.shape)

    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 0.2, random_state =random_seed)
    #print("Shape of X_train: ", X_train.shape)
    #print("Shape of X_test: ", X_test.shape)


    #print("X_test[:,6] ", X_test[:, 0])
    #print("t_test: ", t_test)
    #plt.scatter(X_train[:, 5], t_train, color = 'red')
    #plt.scatter(X_test[:,0], t_test, color = 'blue')
    #plt.show()

    return X_train, X_test, t_train, t_test

def standardize(X_train, X_test):
    X_train_norm = sc.fit_transform(X_train)
    X_test_norm = sc.transform(X_test)
    return X_train_norm, X_test_norm

def train_predictor_sgd(X_train_norm_ones, t_train_col):
    w = np.zeros((1,31))
    X_train_norm_bar = X_train_norm_ones.transpose()
    #print("w transposed shape = ", w.shape)
    #print("size of X_training_norm_bar = ", X_train_norm_bar.shape)

    #print("X_train_norm_bar col 2 = ", X_train_norm_bar[:,2])

    for epo in range(22):
        for i in range(455):
            #print("############# ROUND ", i, " ################")
            # z size should be (455, 21)
            # step 1: z = wTx(bar)
            z = w.dot(X_train_norm_bar[:,i])
            #print("Size of z = ", z.shape)
            #print("z = ", z)    # each entry is the z for each sample

            # apply the sigmoid function to z
            y = 1 / (1 + np.exp(-z))
            #print("size of y (sig(z)) = ", y.size)
            #print("y = ", y)

            # to be able to use w(t) = w(t−1) − α/N * X_train_norm_bar * (y − t)
            # y needs to be (samples, 1) = (455,1)

            # use w update formula - SGD
            #print("t_train_col[i] = ", t_train_col[i])

            #print("y - t = ", y - t_train_col[i])
            
            cost = np.array((y - t_train_col[i])*X_train_norm_bar[:,i])
            #print("cost = ", cost)
            #print("w trans = ", w.transpose())

            #print("cost ready = ", (0.001*cost).reshape(31,1))
            # updated parameter vector 
            w_updated = w.transpose() - (0.001*cost).reshape(31,1)

            #print("w updated = ", w_updated)

            w = w_updated.transpose()

    return w_updated


# def mod_train_predictor_bgd(X_train_norm_ones, t_train_col):
#     # assume w0 is all 0s

#     # create column vector of 31 zeros for w
#     w = np.zeros((31,1))
#     #print("w_trans = ", w)
#     # transpose the X matrix to get X_bar
#     #X_train_norm_bar = X_train_norm_ones.transpose()
#     #print("w transposed shape = ", w.shape)
#     #print("size of X_training_norm_bar = ", X_train_norm_bar.shape)

#     for i in range(100):
#         #print("############# ROUND ", i, " ################")
#         # z size should be (455, 21)
#         # step 1: z = wTx(bar)
#         z = X_train_norm_ones.dot(w)
#         #print("Size of z = ", z.shape)
#         #print("z = ", z)    # each entry is the z for each sample

#         # apply the sigmoid function to z
#         y = 1 / (1 + np.exp(-z))
#         #rint("size of y (sig(z)) = ", y.size)
#         #print("y = ", y)

#         # to be able to use w(t) = w(t−1) − α/N * X_train_norm_bar * (y − t)
#         # y needs to be (samples, 1) = (455,1)

#         #y_col = y.transpose()

#         # use w update formula - BGD

#         #print("size of y_col = ", y_col.shape)
#         #print("size of t_train = ", t_train_col.shape)

#         cost = X_train_norm_ones.transpose().dot(y - t_train_col) / 455
#         #print("cost = ", cost)

#         # updated parameter vector 
#         w_updated = w - 0.001*cost

#         #print("w updated = ", w_updated)

#         w = w_updated

#     return w_updated



def train_predictor_bgd(X_train_norm_ones, t_train_col):
    # assume w0 is all 0s

    # create column vector of 31 zeros for w
    w = np.zeros((1,31))
    #print("w_trans = ", w)
    # transpose the X matrix to get X_bar
    X_train_norm_bar = X_train_norm_ones.transpose()
    #print("w transposed shape = ", w.shape)
    #print("size of X_training_norm_bar = ", X_train_norm_bar.shape)

    iterations = 1000
    cost_array = np.empty(iterations)

    for i in range(iterations):
        #print("############# ROUND ", i, " ################")
        # z size should be (455, 21)
        # step 1: z = wTx(bar)
        z = w.dot(X_train_norm_bar)
        #print("Size of z = ", z.shape)
        #print("z = ", z)    # each entry is the z for each sample

        # apply the sigmoid function to z
        y = 1 / (1 + np.exp(-z))
        #rint("size of y (sig(z)) = ", y.size)
        #print("y = ", y)

        # to be able to use w(t) = w(t−1) − α/N * X_train_norm_bar * (y − t)
        # y needs to be (samples, 1) = (455,1)

        y_col = y.transpose()

        # use w update formula - BGD

        #print("size of y_col = ", y_col.shape)
        #print("size of t_train = ", t_train_col.shape)

        cost = X_train_norm_bar.dot(y_col - t_train_col) / 455
        #print("cost mean = ", cost.mean())
        #cost_array[i] = cost.mean()

        # updated parameter vector 
        w_updated = w.transpose() - 0.01*cost

        #print("w updated = ", w_updated)

        w = w_updated.transpose()

    return w_updated, cost_array

    
#def predict(w, x_bar):
    
    # z = w_trans.dot(X_train_norm_bar)
    # print("Size of z = ", z.shape)
    # print("z = ", z)    # each entry is the z for each sample

    # apply the sigmoid function to z
    #y = 1 / (1 + np.exp(-(w_trans.dot(X_train_norm_bar))))


def predict(X, w):
    # create column vector of 31 zeros for w
    # print("################################")
    # print("w = ", w)
    # print("X = ", X)
    # transpose the X matrix to get X_bar
    w_trans = w.transpose()
    X_bar = X.transpose()

    # print("################################")
    # print("w_trans = ", w_trans)
    # print("X_bar = ", X_bar)

    #print("w transposed shape = ", w.shape)
    #print("size of X_training_norm_bar = ", X_train_norm_bar.shape)

    iterations = 114
    prediction = np.empty((1,iterations))

    z = w_trans.dot(X_bar)
    #print("Size of z = ", z.shape)
    #print("z = ", z)    # each entry is the z for each sample

    # apply the sigmoid function to z
    #prediction = 1 / (1 + np.exp(-z))

    #print("y = ", prediction[i])

    return z

def classify(predictions, threshold):

    predictions_classified = np.empty(114)
    for i in range(114):
        if predictions[0,i] >= threshold:
            predictions_classified[i] = 1
        else:
            predictions_classified[i] = 0

    return predictions_classified

def misclassification(pred_classified, t_test):
    
    correct_predictions = 0
    #print("predicited class vs. target class -- MISS")
    for i in range(114):
        if pred_classified[i] != t_test[i]:
            correct_predictions = correct_predictions + 1
            #print("sample = ", i, " -- ", pred_classified[i], " " , t_test[i])

    
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0

    for i in range(114):
        if t_test[i] == 1.0:
            if pred_classified[i] == 1:
                # true positive
                true_pos = true_pos + 1
            else:
                # false negative
                false_neg = false_neg + 1
        else: 
            if pred_classified[i] == 1:
                # false positive
                false_pos = false_pos + 1
            else:
                # true negative 
                true_neg = true_neg + 1

    misclass_rate = (false_neg + false_pos)/114
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    false_pos_rate = false_pos / (false_pos + true_neg)
    print("precision = ", precision)
    print("recall = ", recall)
    f1 = 2 / ((1/precision) + (1/recall))
    return misclass_rate, f1, precision, recall, false_pos_rate

def PR_ROC(predictions_bgd, predictions_sgd, t_test):
        
    # different classifiers for bgd
    sorted_pred_bgd = np.sort(predictions_bgd[0])
    precision_bgd = np.empty(114)
    recall_bgd = np.empty(114)
    fp_rate_bgd = np.empty(114)

    sorted_pred_sgd = np.sort(predictions_sgd[0])
    precision_sgd = np.empty(114)
    recall_sgd = np.empty(114)
    fp_rate_sgd = np.empty(114)


    for i in range(len(sorted_pred_bgd)):

        batch_z = sorted_pred_bgd[i]
        stoch_z = sorted_pred_sgd[i]
        print("################ ", batch_z, " ##################")
        predictions_classified_bgd = classify(predictions_bgd, batch_z)
        predictions_classified_sgd = classify(predictions_sgd, stoch_z)

        misclass_rate_bgd, f1_bgd, prec, recall, fp_rate = misclassification(predictions_classified_bgd, t_test)
        precision_bgd[i] = prec
        recall_bgd[i] = recall
        fp_rate_bgd[i] = fp_rate
        print("misclassifiction rate BGD= ", misclass_rate_bgd, "   f1 = ", f1_bgd)

        misclass_rate_sgd, f1_sgd, prec, recall, fp_rate = misclassification(predictions_classified_sgd, t_test)
        precision_sgd[i] = prec
        recall_sgd[i] = recall
        fp_rate_sgd[i] = fp_rate
        print("misclassifiction rate SGD= ", misclass_rate_sgd, "   f1 = ", f1_sgd)

    ################ PR ##################
    plt.plot(recall_bgd, precision_bgd, "b:x")
    plt.title("Batch Gradient Decent PR Curve")
    plt.show()

    plt.plot(recall_sgd, precision_sgd, "r:x")
    plt.title("Stoch. Gradient Decent PR Curve")
    plt.show()

    ################# ROC ##################
    plt.plot(fp_rate_bgd, recall_bgd, "b:x")
    plt.title("Batch Gradient Decent ROC Curve")
    plt.show()

    plt.plot(fp_rate_sgd, recall_sgd, "r:x")
    plt.title("Stoch. Gradient Decent ROC Curve")
    plt.show()

def scikit_implementation(X_train, X_test, t_train, t_test):
    scikit_model = LogisticRegression()
    scikit_model.fit(X_train, t_train)
    #scikit_model.predict(X_test[0:10])



def main():
    print("Assignment 2")

    # fetch data 
    X_train, X_test, t_train, t_test = fetch_data()
    X_train_norm, X_test_norm = standardize(X_train, X_test)

    #print("X_train_norm shape = ", X_train_norm.shape)

    # # add ones column to X
    # ones_col = np.ones((455,1))
    # X_train_norm_ones = np.concatenate((ones_col, X_train_norm), axis=1)
    # t_train_col = np.array(t_train).reshape(455,1)  # needs to be converted to numpy array

    # #print("X_train_norm_ones: ", X_train_norm_ones)

    # w_bgd, costs = train_predictor_bgd(X_train_norm_ones, t_train_col)

    # w_sgd = train_predictor_sgd(X_train_norm_ones, t_train_col)

    # print("Batch GD parameters: ", w_bgd)

    # print("Sto. GD parameters: ", w_sgd)

    # # add ones column to X_test data
    # ones_col_test = np.ones((114,1))
    # X_test_norm_ones = np.concatenate((ones_col_test, X_test_norm), axis=1)

    # predictions_bgd = predict(X_test_norm_ones, w_bgd)
    # predictions_sgd = predict(X_test_norm_ones, w_sgd)

    # PR_ROC(predictions_bgd, predictions_sgd, t_test)

    ### scikits implementation ###
    scikit_implementation(X_train, X_test, t_train, t_test)

    

    
main()


##### MOD and normal batch gradient decent give the same parameters