# Logistic Regression for Binary Classification

import numpy as np
from numpy import random
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier


sc = StandardScaler()

random_seed = 2726

def fetch_data():
    breast_cancer = load_breast_cancer()
    print(breast_cancer.DESCR)
    X, t = load_breast_cancer(return_X_y=True)
    X_train, X_test, t_train, t_test = train_test_split(X, t, test_size = 0.2, random_state =random_seed)

    return X_train, X_test, t_train, t_test

def standardize(X_train, X_test):
    X_train_norm = sc.fit_transform(X_train)
    X_test_norm = sc.transform(X_test)
    return X_train_norm, X_test_norm

def train_predictor_sgd(X_train_norm_ones, t_train_col, alpha):
    w = np.zeros((1,31))
    X_train_norm_bar = X_train_norm_ones.transpose()

    epoch = 11
    cost_array = np.empty(epoch)
    costs_epoch = np.empty(455)

    for epo in range(epoch):
        for i in range(455):
            #print("############# ROUND ", i, " ################")
            # step 1: z = wTx(bar)
            z = w.dot(X_train_norm_bar[:,i])

            # apply the sigmoid function to z
            y = 1 / (1 + np.exp(-z))
            
            cost = np.array((y - t_train_col[i])*X_train_norm_bar[:,i])
            costs_epoch[i] = np.average(cost)
            w_updated = w.transpose() - (alpha*cost).reshape(31,1)

            #print("w updated = ", w_updated)
            w = w_updated.transpose()
        cost_array[epo] = np.average(costs_epoch)

    return w_updated, cost_array

def train_predictor_bgd(X_train_norm_ones, t_train_col, alpha):
    # assume w0 is all 0s

    # create column vector of 31 zeros for w
    w = np.zeros((1,31))
    # transpose the X matrix to get X_bar
    X_train_norm_bar = X_train_norm_ones.transpose()

    iterations = 5000
    cost_array = np.empty(iterations)

    for i in range(iterations):
        # step 1: z = wTx(bar)
        z = w.dot(X_train_norm_bar)

        # apply the sigmoid function to z
        y = 1 / (1 + np.exp(-z))

        y_col = y.transpose()

        # use w update formula - BGD
        cost = X_train_norm_bar.dot(y_col - t_train_col) / 455
        cost_array[i] = np.average(cost)
        # updated parameter vector 
        w_updated = w.transpose() - alpha*cost

        w = w_updated.transpose()

    return w_updated, cost_array


def predict(X, w):
    # transpose the X matrix to get X_bar
    w_trans = w.transpose()
    X_bar = X.transpose()

    iterations = 114
    prediction = np.empty((1,iterations))

    z = w_trans.dot(X_bar)

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
    for i in range(114):
        if pred_classified[i] != t_test[i]:
            correct_predictions = correct_predictions + 1

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
    f1 = 2 / ((1/precision) + (1/recall))
    return misclass_rate, f1, precision, recall, false_pos_rate


def PR_ROC(predictions_bgd, predictions_sgd, t_test, title):

    # different classifiers for bgd
    sorted_pred_bgd = np.sort(predictions_bgd[0])
    precision_bgd = np.empty(114)
    recall_bgd = np.empty(114)
    fp_rate_bgd = np.empty(114)
    f1_val_bgd = np.empty(114)
    missclass_bgd = np.empty(114)

    sorted_pred_sgd = np.sort(predictions_sgd[0])
    precision_sgd = np.empty(114)
    recall_sgd = np.empty(114)
    fp_rate_sgd = np.empty(114)
    f1_val_sgd = np.empty(114)
    missclass_sgd = np.empty(114)


    for i in range(len(sorted_pred_bgd)):

        batch_z = sorted_pred_bgd[i]
        stoch_z = sorted_pred_sgd[i]
        #print("################ ", batch_z, " ##################")
        predictions_classified_bgd = classify(predictions_bgd, batch_z)
        predictions_classified_sgd = classify(predictions_sgd, stoch_z)

        misclass_rate_bgd, f1_bgd, prec, recall, fp_rate = misclassification(predictions_classified_bgd, t_test)
        precision_bgd[i] = prec
        recall_bgd[i] = recall
        fp_rate_bgd[i] = fp_rate
        f1_val_bgd[i] = f1_bgd
        missclass_bgd[i] = misclass_rate_bgd
        #print("misclassifiction rate BGD= ", misclass_rate_bgd, "   f1 = ", f1_bgd)

        misclass_rate_sgd, f1_sgd, prec, recall, fp_rate = misclassification(predictions_classified_sgd, t_test)
        precision_sgd[i] = prec
        recall_sgd[i] = recall
        fp_rate_sgd[i] = fp_rate
        f1_val_sgd[i] = f1_sgd
        missclass_sgd[i] = misclass_rate_sgd
        #print("misclassifiction rate SGD= ", misclass_rate_sgd, "   f1 = ", f1_sgd)

    plt.suptitle(title)
    plt.subplot(2,1,1)
    ################ PR ##################
    plt.plot(recall_bgd, precision_bgd, "b:x")
    plt.title("Batch Gradient Decent PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.subplot(2,1,2)
    plt.plot(recall_sgd, precision_sgd, "r:x")
    plt.title("Stoch. Gradient Decent PR Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.show()


    plt.suptitle(title)
    plt.subplot(2,1,1)
    ################# ROC ##################
    plt.plot(fp_rate_bgd, recall_bgd, "b:x")
    plt.title("Batch Gradient Decent ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.subplot(2,1,2)
    plt.plot(fp_rate_sgd, recall_sgd, "r:x")
    plt.title("Stoch. Gradient Decent ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.show()

    return missclass_bgd[46], missclass_sgd[46], f1_val_bgd[46], f1_val_sgd[46]
    

def scikit_implementation_sgd(X_train, X_test, t_train, t_test):

    sgd_pred = SGDClassifier(loss='log_loss' , max_iter=5000, random_state=random_seed)
    sgd_pred.fit(X_train, t_train)
    print(sgd_pred.coef_)
    # score the model on the test data
    score = sgd_pred.score(X_test, t_test)
    print("Scikit Implementation Score = ", score)

    return np.array(sgd_pred.coef_).reshape(31,1)

def scikit_implementation_bgd(X_train, X_test, t_train, t_test):

    bgd_pred = LogisticRegression()
    # train the model
    bgd_pred.fit(X_train, t_train)

    print(bgd_pred.coef_)
    # score the model on the test data
    score = bgd_pred.score(X_test, t_test)
    print("Scikit Implementation Score = ", score)

    return np.array(bgd_pred.coef_).reshape(31,1)




def main():
    print("Assignment 2")

    # fetch data 
    X_train, X_test, t_train, t_test = fetch_data()
    X_train_norm, X_test_norm = standardize(X_train, X_test)

    # add ones column to X
    ones_col = np.ones((455,1))
    X_train_norm_ones = np.concatenate((ones_col, X_train_norm), axis=1)
    t_train_col = np.array(t_train).reshape(455,1)  # needs to be converted to numpy array

    # train the predictor 
    w_bgd_a1, costs_a1 = train_predictor_bgd(X_train_norm_ones, t_train_col, 0.0000001)
    w_bgd_a2, costs_a2 = train_predictor_bgd(X_train_norm_ones, t_train_col, 0.0001)
    w_bgd, costs_a3 = train_predictor_bgd(X_train_norm_ones, t_train_col, 0.001)

    # plot the learning curves bgd
    plt.plot(costs_a1, "b")
    plt.plot(costs_a2, "g")
    plt.plot(costs_a3, "r")
    plt.title("Learning Curve of Different Alphas - Batch Gradient Decent ")
    plt.legend(['alpha1 = 0.0000001', 'alpha2 = 0.0001', 'alpha3 = 0.001'])
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    plt.show()

    # plot the learning curves sgd
    w_sgd_a1, costs_a1 = train_predictor_sgd(X_train_norm_ones, t_train_col, 0.0000001)
    w_sgd_a2, costs_a2 = train_predictor_sgd(X_train_norm_ones, t_train_col, 0.0001)
    w_sgd, costs_a3 = train_predictor_sgd(X_train_norm_ones, t_train_col, 0.001)

    plt.plot(costs_a1, "b")
    plt.plot(costs_a2, "g")
    plt.plot(costs_a3, "r")
    plt.title("Learning Curve of Different Alphas - Stochastic Gradient Decent ")
    plt.legend(['alpha1 = 0.0000001', 'alpha2 = 0.0001', 'alpha3 = 0.001'])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.show()


    print("Batch GD parameters: ", w_bgd)
    for i in range(31):
        print(w_bgd[i,0], ", ", end='')

    print("Sto. GD parameters: ", w_sgd)
    for i in range(31):
        print(w_sgd[i,0], ", ", end='')

    # add ones column to X_test data
    ones_col_test = np.ones((114,1))
    X_test_norm_ones = np.concatenate((ones_col_test, X_test_norm), axis=1)

    # use model to predict on the test set
    predictions_bgd = predict(X_test_norm_ones, w_bgd)
    predictions_sgd = predict(X_test_norm_ones, w_sgd)

    # classify and create PR and ROC curves
    missclass_bgd, missclass_sgd, f1_bgd, f1_sgd = PR_ROC(predictions_bgd, predictions_sgd, t_test, "Nicole's Logistic Regression Implementation")

    ### scikits implementation ###
    print("############## SCIKIT ################")
    sk_bgd_coeff = scikit_implementation_bgd(X_train_norm_ones, X_test_norm_ones, t_train_col, t_test)
    sk_sgd_coeff = scikit_implementation_sgd(X_train_norm_ones, X_test_norm_ones, t_train_col, t_test)

    predictions_bgd_scikit = predict(X_test_norm_ones, sk_bgd_coeff)
    predictions_sgd_scikit = predict(X_test_norm_ones, sk_sgd_coeff)

    missclass_bgd_sk, missclass_sgd_sk, f1_bgd_sk, f1_sgd_sk = PR_ROC(predictions_bgd_scikit, predictions_sgd_scikit, t_test, "Scikit's Logistic Regression Implementation")

    # print metrics
    print("Nicole's Implementation")
    print("Misclass rate bgd = ", missclass_bgd)
    print("Misclass rate sgd = ", missclass_sgd)
    print("f1 bgd = ", f1_bgd)
    print("f1 sgd = ", f1_sgd)

    print("Scikit's Implementation")
    print("Misclass rate bgd = ", missclass_bgd_sk)
    print("Misclass rate sgd = ", missclass_sgd_sk)
    print("f1 bgd = ", f1_bgd_sk)
    print("f1 sgd = ", f1_sgd_sk)


main()


