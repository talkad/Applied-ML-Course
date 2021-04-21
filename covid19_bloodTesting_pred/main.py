import pandas as pd
from imblearn.over_sampling import SVMSMOTE
from imblearn.metrics import specificity_score, sensitivity_score
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from xgboost import XGBClassifier
import numpy as np
from copy import deepcopy
import warnings
import pickle
from scipy import stats
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import shap

RBC_IDX = 3
HGB_IDX = 1
WBC_IDX = 7
LYM_IDX = 4
MONO_IDX = 13
NEU_IDX = 15


def nested_cross_validation(model, hyper_params, X_features, y_labels, ratio_flag):
    k = 5
    best_f1 = 0
    best_hyperParams = {}

    cv_outer = KFold(n_splits=k, shuffle=True)
    # enumerate splits
    outer_results = list()
    for train_ix, test_ix in cv_outer.split(X_features):

        X_train, X_test = X_features[train_ix, :], X_features[test_ix, :]
        y_train, y_test = y_labels[train_ix], y_labels[test_ix]

        cv_inner = KFold(n_splits=k, shuffle=True)

        new_model = deepcopy(model)

        imputer = IterativeImputer(max_iter=250)
        imputer.fit(X_train)  # fit the imputer only to training set
        X_train = imputer.transform(X_train)
        X_test = imputer.transform(X_test)

        oversample = SVMSMOTE(k_neighbors=5)
        X_train, y_train = oversample.fit_resample(X_train, y_train)

        if ratio_flag:
            X_train = add_ratio_cols(X_train)
            X_test = add_ratio_cols(X_test)

        search = GridSearchCV(new_model, hyper_params, scoring='f1_macro', cv=cv_inner, refit=True)
        result = search.fit(X_train, y_train)
        best_model = result.best_estimator_

        yhat = best_model.predict(X_test)
        f1 = f1_score(y_test, yhat, average='macro')

        sen = sensitivity_score(y_test, yhat)
        spec = specificity_score(y_test, yhat)


        if f1 > best_f1:
            best_f1 = f1
            best_hyperParams = result.best_params_

        outer_results.append(accuracy_score(y_test, yhat))
        print(
            f'accuracy: {accuracy_score(y_test, yhat)}  |  f1: {f1_score(y_test, yhat, average="macro")}  |  sensetivity: {sen}  |  specificity: {spec}  |  AUROC: {roc_auc_score(y_test, yhat)}')

    # print(f'mean acc: {np.mean(outer_results)} |  best f1: {best_f1} | params: {best_hyperParams}')
    return best_hyperParams


def add_ratio_cols(X_features):
    return np.c_[X_features, stats.zscore(X_features[:, HGB_IDX] / X_features[:, RBC_IDX]),
                 stats.zscore(X_features[:, WBC_IDX] / X_features[:, RBC_IDX]),
                 stats.zscore(X_features[:, LYM_IDX] / X_features[:, RBC_IDX]),
                 stats.zscore(X_features[:, MONO_IDX] / X_features[:, RBC_IDX]),
                 stats.zscore(X_features[:, NEU_IDX] / X_features[:, RBC_IDX])]


def train_best_model(model, X_features, y_labels, ratio_flag):
    acc_list = list()
    f1_list = list()
    sens_list = list()
    spec_list = list()
    auroc_list = list()

    best_acc = 0
    best_model = None
    data = []

    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_features, y_labels, test_size=0.2)  # choose 20% randomly

        imputer = IterativeImputer(max_iter=500)
        imputer.fit(X_train)  # fit the imputer only to training set
        X_train = imputer.transform(X_train)
        X_test = imputer.transform(X_test)

        oversample = SVMSMOTE(k_neighbors=5)
        X_train, y_train = oversample.fit_resample(X_train, y_train)

        if ratio_flag:
            X_train = add_ratio_cols(X_train)
            X_test = add_ratio_cols(X_test)

        current_model = deepcopy(model)
        current_model.fit(X_train, y_train)

        yhat = current_model.predict(X_test)

        acc = accuracy_score(y_test, yhat)
        f1 = f1_score(y_test, yhat, average="macro")
        sens = sensitivity_score(y_test, yhat)
        spec = specificity_score(y_test, yhat)
        auroc = roc_auc_score(y_test, yhat)

        acc_list.append(acc)
        f1_list.append(f1)
        sens_list.append(sens)
        spec_list.append(spec)
        auroc_list.append(auroc)

        if acc > best_acc:
            data = [X_train, X_test, y_train, y_test]
            best_acc = acc
            best_model = current_model

    print('\nMean accuracy: %.3f  |  Mean f1: %.3f  |  Mean sensitivity: %.3f  |  Mean specificity: %.3f  |  Mean AUROC: %.3f'
          % (np.mean(acc_list), np.mean(f1_list), np.mean(sens_list), np.mean(spec_list), np.mean(auroc_list)))
    return best_model, data


def store_model(model, filename):
    pickle_file = open(filename, 'wb')
    pickle.dump(model, pickle_file)
    pickle_file.close()


def load_model(filename):
    pickle_file = open(filename, 'rb')
    loaded = pickle.load(pickle_file)
    pickle_file.close()

    return loaded


def show_shap_plot(modeldata_filename):
    feature_lst = ["Hematocrit", "Hemoglobin", "Platelets", "Red blood Cells", "Lymphocytes",
                   "MCH", "MCHC",
                   "Leukocytes", "Basophils", "Eosinophils", "Lactic Dehydrogenase", "MCV",
                   "RDW", "Monocytes", "Mean platelet volume ", "Neutrophils",
                   "Proteina C reativa mg/dL", "Creatinine", "Urea", "Potassium", "Sodium", "Aspartate transaminase",
                   "Alanine transaminase"]
    [X_train, X_test, y_train, y_test, model] = load_model(modeldata_filename)
    explainer = shap.Explainer(model.predict, X_train, feature_names=feature_lst)
    shap_values = explainer(X_test)
    shap.plots.beeswarm(shap_values, max_display=23, plot_size=(22, 10))

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

        # 2b + 3
        for append_ratio in [False, True]:
            # Logistic Regression
            print(f'Logistic Regression | with ratio={append_ratio}: ')
            space = dict()
            space['solver'] = ['liblinear', 'saga']
            model_LR = LogisticRegression(random_state=1, max_iter=1000, dual=False)
            best_hyper_params = nested_cross_validation(model_LR, space, X, y, ratio_flag=append_ratio)

            model_LR = LogisticRegression(random_state=1, max_iter=1000, dual=False, solver=best_hyper_params['solver'])
            model_LR, data_LR = train_best_model(model_LR, X, y, ratio_flag=append_ratio)
            print(f'Best hyper params: {best_hyper_params}\n')

            data_LR.append(model_LR)
            store_model(data_LR, f'LR_{append_ratio}.p')

            # Random Forest
            print(f'Random Forest: | with ratio={append_ratio}: ')
            space = dict()
            space['n_estimators'] = [10, 20, 30, 40] + list(range(50, 105, 5))
            space['max_depth'] = [2, 4, 8, 16, 32, 64]

            model_RF = RandomForestClassifier()
            best_hyper_params = nested_cross_validation(model_RF, space, X, y, ratio_flag=append_ratio)

            model_RF = RandomForestClassifier(n_estimators=best_hyper_params['n_estimators'],
                                              max_depth=best_hyper_params['max_depth'])
            model_RF, data_RF = train_best_model(model_RF, X, y, ratio_flag=append_ratio)
            print(f'Best hyper params: {best_hyper_params}\n')

            data_RF.append(model_RF)
            store_model(data_RF, f'RF_{append_ratio}.p')

            # XGBoost
            print(f'XGBoost: | with ratio={append_ratio}: ')
            space = dict()
            space['n_estimators'] = [10, 20, 30, 40] + list(range(50, 105, 5))
            space['max_depth'] = [2, 4, 8, 16, 32, 64]
            space['learning_rate'] = [0.1, 0.05, 0.01]

            model_XGB = XGBClassifier(objective='binary:logistic', eval_metric='logloss')
            best_hyper_params = nested_cross_validation(model_XGB, space, X, y, ratio_flag=append_ratio)

            model_XGB = XGBClassifier(objective='binary:logistic', n_estimators=best_hyper_params['n_estimators'],
                                      max_depth=best_hyper_params['max_depth'],
                                      learning_rate=best_hyper_params['learning_rate'], eval_metric='logloss')
            model_XGB, data_XGB = train_best_model(model_XGB, X, y, ratio_flag=append_ratio)
            print(f'Best hyper params: {best_hyper_params}\n')

            data_XGB.append(model_XGB)
            store_model(data_XGB, f'XGB_{append_ratio}.p')

        # CatBoost
        print(f'CatBoost: ')

        space = dict()
        space['n_estimators'] = [10, 20, 30, 40] + list(range(50, 105, 5))
        space['max_depth'] = [2, 4, 8]
        space['learning_rate'] = [0.1, 0.05, 0.01]

        model_CatBoost = CatBoostClassifier()
        best_hyper_params = nested_cross_validation(model_CatBoost, space, X, y, ratio_flag=False)

        model_CatBoost = CatBoostClassifier(learning_rate=best_hyper_params['learning_rate'])
        model_CatBoost, data_Cat = train_best_model(model_CatBoost, X, y, ratio_flag=False)
        print(f'Best hyper params: {best_hyper_params}\n')

        data_Cat.append(model_CatBoost)
        store_model(data_Cat, f'CatBoost.p')

        # LightGBM
        print(f'LightGBM:')

        space = dict()
        space['n_estimators'] = [10, 20, 30, 40] + list(range(50, 105, 5))
        space['max_depth'] = [2, 4, 8, 16, 32, 64]
        space['learning_rate'] = [0.1, 0.05, 0.01]

        model_GBM = LGBMClassifier(objective='binary')
        best_hyper_params = nested_cross_validation(model_GBM, space, X, y, ratio_flag=False)

        model_GBM = LGBMClassifier(objective='binary', n_estimators=best_hyper_params['n_estimators'],
                                   max_depth=best_hyper_params['max_depth'],
                                   learning_rate=best_hyper_params['learning_rate'])
        model_GBM, data_GBM = train_best_model(model_GBM, X, y, ratio_flag=False)
        print(f'Best hyper params: {best_hyper_params}\n')

        data_GBM.append(model_GBM)
        store_model(data_GBM, f'LightGBM.p')

        # SHAP
        show_shap_plot('RF_False.p')
        show_shap_plot('XGB_False.p')
        show_shap_plot('LightGBM.p')
        show_shap_plot('CatBoost.p')
