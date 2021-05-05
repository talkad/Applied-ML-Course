from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss


def train():
    model_list = list()
    for depth in range(2, 16):
        model = DecisionTreeClassifier(max_depth=depth)
        model.fit(X_train, y_train)
        model_list.append(model)

    return model_list


def predict(model_list, x):

    for index in range(2, 16):
        model = model_list[index]
        pred = model.predict([x])

        # print(f'iteration {index} with confidence {model.predict_proba([X_test[test]])[0][y_hat][0]}')

        if model.predict_proba([x])[0][pred][0] > threshold:
            # print(f'predicted class: {y_hat} with probability {model.predict_proba([X_test[test]])[0][y_hat][0]}')
            return pred, model.predict_proba([x])[0]

    pred = models[-1].predict([x])
    return pred, models[-1].predict_proba([x])[0]


if __name__ == '__main__':
    threshold = 0.95

    digits = load_digits()

    X_train, X_test, y_train, y_test = train_test_split(digits.images, digits.target, test_size=0.2)

    plt.gray()
    plt.matshow(X_test[256])
    print(f'actual class: {y_test[256]}')
    plt.show()

    X_train = [X.flatten() for X in X_train]
    X_test = [X.flatten() for X in X_test]

    # cascade classifiers training
    models = train()

    # evaluate model
    y_hat = list()

    for i in range(len(y_test)):
        _, probability = predict(models, X_test[i])
        y_hat.append(probability)

    print(f'log loss = {log_loss(y_test, y_hat)}')

    print(f'prediction: {predict(models, X_test[256])}')

