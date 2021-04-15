import pandas as pd
from imblearn.over_sampling import SVMSMOTE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from xgboost import XGBClassifier
from numpy import mean
from copy import deepcopy
import warnings
import pickle


def nested_cross_validation(model, hyper_params):
    k = 5
    best_acc = 0
    best_hyperParams = {}

    cv_outer = KFold(n_splits=k, shuffle=True, random_state=1)
    # enumerate splits
    outer_results = list()
    for train_ix, test_ix in cv_outer.split(X):
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        cv_inner = KFold(n_splits=k, shuffle=True, random_state=1)

        model = deepcopy(model)

        imputer = IterativeImputer(max_iter=250)
        imputer.fit(X_train)  # fit the imputer only to training set
        X_train = imputer.transform(X_train)
        X_test = imputer.transform(X_test)

        oversample = SVMSMOTE(k_neighbors=5)
        X_train, y_train = oversample.fit_resample(X_train, y_train)

        search = GridSearchCV(model, hyper_params, scoring='f1', cv=cv_inner, refit=True)
        result = search.fit(X_train, y_train)
        best_model = result.best_estimator_

        yhat = best_model.predict(X_test)
        acc = accuracy_score(y_test, yhat)

        if acc > best_acc:
            best_acc = acc
            best_hyperParams = result.best_params_

        outer_results.append(acc)

    print(f'acc: {best_acc} | params: {best_hyperParams}')
    return best_hyperParams


def train_best_model(model):
    acc_list = list()
    best_acc = 0
    best_model = None

    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        imputer = IterativeImputer(max_iter=250)
        imputer.fit(X_train)  # fit the imputer only to training set
        X_train = imputer.transform(X_train)
        X_test = imputer.transform(X_test)

        oversample = SVMSMOTE(k_neighbors=5)
        X_train, y_train = oversample.fit_resample(X_train, y_train)

        current_model = deepcopy(model)
        current_model.fit(X_train, y_train)

        yhat = current_model.predict(X_test)
        acc = accuracy_score(y_test, yhat)

        acc_list.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_model = current_model

    print('Mean accuracy: %.3f' % (mean(acc_list)))
    return best_model


def store_model(model, filename):
    pickle_file = open(filename, 'wb')
    pickle.dump(model, pickle_file)
    pickle_file.close()


def load_model(filename):
    pickle_file = open(filename, 'rb')
    loaded = pickle.load(pickle_file)
    pickle_file.close()

    return loaded


if __name__ == "__main__":
    warnings.simplefilter("ignore")  # ignore ConvergenceWarning

    # 2a - performing pre-processing
    df = pd.read_excel("dataset.xlsx", engine="openpyxl")

    df = df[["SARS-Cov-2 exam result", "Hematocrit", "Hemoglobin", "Platelets", "Red blood Cells", "Lymphocytes",
             "Mean corpuscular hemoglobin (MCH)", "Mean corpuscular hemoglobin concentration (MCHC)",
             "Leukocytes", "Basophils", "Eosinophils", "Lactic Dehydrogenase", "Mean corpuscular volume (MCV)",
             "Red blood cell distribution width (RDW)", "Monocytes", "Mean platelet volume ", "Neutrophils",
             "Proteina C reativa mg/dL", "Creatinine", "Urea", "Potassium", "Sodium", "Aspartate transaminase",
             "Alanine transaminase"]]

    df.dropna(thresh=23 * 0.05, inplace=True)  # filtering rows with more than one empty cell (~95% threshold)

    covid_results = df["SARS-Cov-2 exam result"]
    df.drop(columns=["SARS-Cov-2 exam result"], inplace=True)

    X = df.to_numpy()
    y = covid_results.apply(lambda res: 0 if res == 'negative' else 1).to_numpy()

    # 2b
    # Logistic Regression
    print('Logistic Regression: ')
    space = dict()
    space['solver'] = ['liblinear', 'saga']
    model_LR = LogisticRegression(random_state=1, max_iter=1000, dual=False)
    best_hyper_params = nested_cross_validation(model_LR, space)

    model_LR = LogisticRegression(random_state=1, max_iter=1000, dual=False, solver=best_hyper_params['solver'])
    model_LR = train_best_model(model_LR)

    info = '_'.join(best_hyper_params.values())
    store_model(model_LR, f'LR_{str(info)}.p')

    # Random Forest
    print('Random Forest: ')
    space = dict()
    space['n_estimators'] = [10, 20, 30, 40] + list(range(50, 105, 5))
    space['max_depth'] = [2, 4, 8, 16, 32, 64]

    model_RF = RandomForestClassifier()
    best_hyper_params = nested_cross_validation(model_RF, space)

    model_RF = RandomForestClassifier(n_estimators=best_hyper_params['n_estimators'], max_depth=best_hyper_params['max_depth'])
    model_RF = train_best_model(model_RF)

    info = '_'.join(str(x) for x in best_hyper_params.values())
    store_model(model_RF, f'RF_{str(info)}.p')

    # XGBoost
    print('XGBoost: ')
    space = dict()
    space['n_estimators'] = [10, 20, 30, 40] + list(range(50, 105, 5))
    space['max_depth'] = [2, 4, 8, 16, 32, 64]
    space['learning_rate'] = [0.1, 0.05, 0.01]

    model_XGB = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
    best_hyper_params = nested_cross_validation(model_XGB, space)

    model_XGB = XGBClassifier(objective='binary:logistic', n_estimators=best_hyper_params['n_estimators'],
                              max_depth=best_hyper_params['max_depth'],
                              learning_rate=best_hyper_params['learning_rate'], eval_metric='logloss')
    model_XGB = train_best_model(model_XGB)

    info = '_'.join(str(x) for x in best_hyper_params.values())
    store_model(model_XGB, f'XGB_{info}.p')

