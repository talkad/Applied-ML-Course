from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import log_loss
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import cifar10

improvement = False


def train(X2, y2):
    model_list = list()
    data = X2

    for depth in range(2, 16):

        if improvement:
            model = LGBMClassifier(max_depth=depth)
        else:
            model = DecisionTreeClassifier(max_depth=depth)

        model.fit(data, y2)
        model_list.append(model)

        if improvement:
            prev_predictions = model_list[depth - 2].predict_proba(data)
            data = np.hstack((X2, prev_predictions))

    return model_list


def predict(model_list, x):

    original_x = x

    for index in range(len(model_list)):

        model = model_list[index]
        pred = model.predict([x])

        if model.predict_proba([x])[0][pred] > threshold:
            return pred, model.predict_proba([x])[0]
        elif improvement:
            x = np.append(original_x, model.predict_proba([x])[0])

    pred = model_list[-1].predict([x])
    return pred, model_list[-1].predict_proba([x])[0]


if __name__ == '__main__':
    threshold = 0.95
    test_ration = 0.1

    # filename = 'frogs_MFCCs.csv'
    #
    # df = pd.read_csv(filename)
    # df = df.sample(frac=1).reset_index(drop=True)
    # df[['Family']] = df[['Family']].apply(lambda col: pd.Categorical(col).codes)
    # y = df['Family']
    # df.drop(columns=['Family', 'Genus', 'Species', 'RecordID'], inplace=True)
    #
    # X = df.to_numpy()
    # total = len(X)
    #
    # X_train, X_test, y_train, y_test = X[:int(total*(1-test_ration))], X[int(total*(1-test_ration)):], y[:int(total*(1-test_ration))], y[int(total*(1-test_ration)):]
    #
    # # cascade classifiers training
    # models = train(X_train, y_train)
    #
    # # evaluate model
    # y_hat = list()
    #
    # for i in range(len(y_test)):
    #     _, probability = predict(models, X_test[i])
    #     y_hat.append(probability)
    #
    # print(f'model name: {filename}\n{"with improvements" if improvement else "no improvements"}\nlog loss = {log_loss(y_test, y_hat)}')


    # filename = 'Dry_Bean_Dataset.xlsx'
    #
    # df = pd.read_excel(filename)
    # df = df.sample(frac=1).reset_index(drop=True)
    # df[['Class']] = df[['Class']].apply(lambda col: pd.Categorical(col).codes)
    # y = df['Class']
    #
    # X = df.to_numpy()
    # total = len(X)
    #
    # X_train, X_test, y_train, y_test = X[:int(total*(1-test_ration))], X[int(total*(1-test_ration)):], y[:int(total*(1-test_ration))], y[int(total*(1-test_ration)):]
    #
    # # cascade classifiers training
    # models = train(X_train, y_train)
    #
    # # evaluate model
    # y_hat = list()
    #
    # for i in range(len(y_test)):
    #     _, probability = predict(models, X_test[i])
    #     y_hat.append(probability)
    #
    # print(f'model name: {filename}\n{"with improvements" if improvement else "no improvements"}\nlog loss = {log_loss(y_test, y_hat)}')


    # filename = 'winequality-red.csv'
    #
    # df = pd.read_csv(filename)
    # df = df.sample(frac=1).reset_index(drop=True)
    # df = df[4 <= df.quality]
    # df = df[df.quality <= 7]
    #
    # df[['quality']] = df[['quality']].apply(lambda col: pd.Categorical(col).codes)
    # y = df['quality']
    #
    # X = df.to_numpy()
    # total = len(X)
    #
    # X_train, X_test, y_train, y_test = X[:int(total*(1-test_ration))], X[int(total*(1-test_ration)):], y[:int(total*(1-test_ration))], y[int(total*(1-test_ration)):]
    #
    # # cascade classifiers training
    # models = train(X_train, y_train)
    #
    # # evaluate model
    # y_hat = list()
    #
    # for i in range(len(y_test)):
    #     _, probability = predict(models, X_test[i])
    #     y_hat.append(probability)
    #
    # print(f'model name: {filename}\n{"with improvements" if improvement else "no improvements"}\nlog loss = {log_loss(y_test, y_hat)}')


    # filename = 'CTG.xls'
    #
    # df = pd.read_excel(filename)
    # df.drop(columns=['b', 'e', 'AC', 'FM', 'UC', 'DL', 'DS', 'DP', 'DR'], inplace=True)
    #
    # df = df.sample(frac=1).reset_index(drop=True)
    # df[['CLASS']] = df[['CLASS']].apply(lambda col: pd.Categorical(col).codes)
    # y = df['CLASS']
    # df.drop(columns=['CLASS'], inplace=True)
    #
    # X = df.to_numpy()
    # total = len(X)
    #
    # X_train, X_test, y_train, y_test = X[:int(total*(1-test_ration))], X[int(total*(1-test_ration)):], y[:int(total*(1-test_ration))], y[int(total*(1-test_ration)):]
    #
    # # cascade classifiers training
    # models = train(X_train, y_train)
    #
    # # evaluate model
    # y_hat = list()
    #
    # for i in range(len(y_test)):
    #     _, probability = predict(models, X_test[i])
    #     y_hat.append(probability)
    #
    # print(f'model name: {filename}\n{"with improvements" if improvement else "no improvements"}\nlog loss = {log_loss(y_test, y_hat)}')


    # filename = 'hmnist.csv'
    #
    # df = pd.read_csv(filename)
    # df = df.sample(frac=1).reset_index(drop=True)
    # df[['label']] = df[['label']].apply(lambda col: pd.Categorical(col).codes)
    # y = df['label']
    #
    # X = df.to_numpy()
    # total = len(X)
    #
    # X_train, X_test, y_train, y_test = X[:int(total * (1 - test_ration))], X[int(total * (1 - test_ration)):], y[:int(
    #     total * (1 - test_ration))], y[int(total * (1 - test_ration)):]
    #
    # # cascade classifiers training
    # models = train(X_train, y_train)
    #
    # print(set(y_test), set(y_train))
    #
    # # evaluate model
    # y_hat = list()
    #
    # for i in range(len(y_test)):
    #     _, probability = predict(models, X_test[i])
    #     y_hat.append(probability)
    #
    # print(
    #     f'model name: {filename}\n{"with improvements" if improvement else "no improvements"}\nlog loss = {log_loss(y_test, y_hat)}')


    # digits = load_digits()
    #
    # X_train, X_test, y_train, y_test = train_test_split(digits.images, digits.target, test_size=test_ration)
    #
    # X_train = [X.flatten() for X in X_train]
    # X_test = [X.flatten() for X in X_test]
    #
    # # cascade classifiers training
    # models = train(X_train, y_train)
    #
    # # evaluate model
    # y_hat = list()
    #
    # for i in range(len(y_test)):
    #     _, probability = predict(models, X_test[i])
    #     y_hat.append(probability)
    #
    # print(f'model name: digits\n{"with improvements" if improvement else "no improvements"}\nlog loss = {log_loss(y_test, y_hat)}')


    # filename = 'letter-recognition.csv'
    #
    # df = pd.read_csv(filename)
    # df = df.sample(frac=1).reset_index(drop=True)
    # df[['class']] = df[['class']].apply(lambda col: pd.Categorical(col).codes)
    # y = df['class']
    # df.drop(columns=['class'], inplace=True)
    #
    # X = df.to_numpy()
    # total = len(X)
    #
    # X_train, X_test, y_train, y_test = X[:int(total*(1-test_ration))], X[int(total*(1-test_ration)):], y[:int(total*(1-test_ration))], y[int(total*(1-test_ration)):]
    #
    # # cascade classifiers training
    # models = train(X_train, y_train)
    #
    # # evaluate model
    # y_hat = list()
    #
    # for i in range(len(y_test)):
    #     _, probability = predict(models, X_test[i])
    #     y_hat.append(probability)
    #
    # print(f'model name: {filename}\n{"with improvements" if improvement else "no improvements"}\nlog loss = {log_loss(y_test, y_hat)}')


    # filename = 'page_block.xlsx'
    #
    # df = pd.read_excel(filename)
    # df = df.sample(frac=1).reset_index(drop=True)
    # df[['class']] = df[['class']].apply(lambda col: pd.Categorical(col).codes)
    # y = df['class']
    # df.drop(columns=['class'], inplace=True)
    #
    # X = df.to_numpy()
    # total = len(X)
    #
    # X_train, X_test, y_train, y_test = X[:int(total*(1-test_ration))], X[int(total*(1-test_ration)):], y[:int(total*(1-test_ration))], y[int(total*(1-test_ration)):]
    #
    # # cascade classifiers training
    # models = train(X_train, y_train)
    #
    # # evaluate model
    # y_hat = list()
    #
    # for i in range(len(y_test)):
    #     _, probability = predict(models, X_test[i])
    #     y_hat.append(probability)
    #
    # print(f'model name: {filename}\n{"with improvements" if improvement else "no improvements"}\nlog loss = {log_loss(y_test, y_hat)}')


    # filename = 'muscles.csv'
    # df = pd.read_csv(filename)
    #
    # df = df.sample(frac=1).reset_index(drop=True)
    # df[["class"]] = df[["class"]].apply(lambda col: pd.Categorical(col).codes)
    # y = df['class']
    # df.drop(columns=['class'], inplace=True)
    #
    # X = df.to_numpy()
    # total = len(X)
    #
    # X_train, X_test, y_train, y_test = X[:int(total*(1-test_ration))], X[int(total*(1-test_ration)):], y[:int(total*(1-test_ration))], y[int(total*(1-test_ration)):]
    #
    # # cascade classifiers training
    # models = train(X_train, y_train)
    #
    # # evaluate model
    # y_hat = list()
    #
    # for i in range(len(y_test)):
    #     _, probability = predict(models, X_test[i])
    #     y_hat.append(probability)
    #
    # print(f'model name: {filename}\n{"with improvements" if improvement else "no improvements"}\nlog loss = {log_loss(y_test, y_hat)}')


    filename = 'segmentation.csv'
    df = pd.read_csv(filename)

    df = df.sample(frac=1).reset_index(drop=True)
    df[["REGIONCENTROIDCOL"]] = df[["REGIONCENTROIDCOL"]].apply(lambda col: pd.Categorical(col).codes)
    y = df['REGIONCENTROIDCOL']
    df.drop(columns=['REGIONCENTROIDCOL'], inplace=True)

    X = df.to_numpy()
    total = len(X)

    X_train, X_test, y_train, y_test = X[:int(total*(1-test_ration))], X[int(total*(1-test_ration)):], y[:int(total*(1-test_ration))], y[int(total*(1-test_ration)):]

    # cascade classifiers training
    models = train(X_train, y_train)

    # evaluate model
    y_hat = list()

    for i in range(len(y_test)):
        _, probability = predict(models, X_test[i])
        y_hat.append(probability)

    print(f'model name: {filename}\n{"with improvements" if improvement else "no improvements"}\nlog loss = {log_loss(y_test, y_hat)}')
