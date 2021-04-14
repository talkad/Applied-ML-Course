import pandas as pd
from imblearn.over_sampling import SVMSMOTE
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from numpy import mean
import warnings


def nested_cross_validation(model, hyper_params):
    return None


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
    # Logistic regression

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

        model = LogisticRegression(random_state=1, max_iter=1000, dual=False)

        # define hyper-parameter space
        space = dict()
        space['solver'] = ['liblinear', 'saga']

        imputer = IterativeImputer(max_iter=250)
        imputer.fit(X_train)  # fit the imputer only to training set
        X_train = imputer.transform(X_train)
        X_test = imputer.transform(X_test)

        oversample = SVMSMOTE(k_neighbors=5)
        X_train, y_train = oversample.fit_resample(X_train, y_train)

        search = GridSearchCV(model, space, scoring='f1', cv=cv_inner, refit=True)
        result = search.fit(X_train, y_train)
        best_model = result.best_estimator_

        yhat = best_model.predict(X_test)
        acc = accuracy_score(y_test, yhat)

        if acc > best_acc:
            best_acc = acc
            best_hyperParams = result.best_params_

        outer_results.append(acc)
        print('> acc=%.3f, est=%.3f, params=%s' % (acc, result.best_score_, result.best_params_))

    # print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))
    print(f'acc: {best_acc} | paarams: {best_hyperParams}')

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

        model_LR = LogisticRegression(random_state=1, max_iter=1000, dual=False, solver=best_hyperParams['solver'])
        model_LR.fit(X_train, y_train)

        yhat = model_LR.predict(X_test)
        acc = accuracy_score(y_test, yhat)

        acc_list.append(acc)
        print(f'{i}) acc: {acc}')
        if acc > best_acc:
            best_acc = acc
            best_model = model_LR

    print('Accuracy: %.3f' % (mean(outer_results)))
