import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from sklearn import linear_model, svm
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from hw1_main import print_feature_importance


def get_xy(df):
    y_filter = ['BUYER_FLAG']
    x = df[df.columns[~df.columns.isin(y_filter)]]
    y = df[y_filter].values.ravel()
    return x, y


def print_evaluation_methods(model, scores, x, y):
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    #model.fit(x, y)
    # todo - complete more functions
    y_pred = cross_val_predict(model, x, y, cv=10)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    print("tp: " + str(tp) + ", " + "fp: " + str(fp) + "\n" +"fn: " + str(fn) + ", " +"tn: " + str(tn))
    tp_profit = 32.7
    fp_profit = -6.05
    fn_profit = 0
    tn_profit = 0
    predict_profit = tn*tn_profit + fn*fn_profit + fp*fp_profit + tp*tp_profit
    print('predict_profit: %.3f' % (predict_profit))


def choose_best_model(df):
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    x, y = get_xy(df)
    print('going to check different models:')
    models = [('RF', RandomForestClassifier()),
              ('LR', LogisticRegression()),
              ('LDA', LinearDiscriminantAnalysis()),
              ('KNN', KNeighborsClassifier()),
              ('CART', DecisionTreeClassifier()),
              ('NB', GaussianNB()),
              ('XGB', XGBClassifier(silent=False, n_jobs=13, random_state=15, n_estimators=100, eval_metric='logloss', class_weight='balanced')),
              ('NN', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))
            ]
    for name, m in models:
        print(f'\nCheck model: {name}:')
        scores = cross_val_score(m, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
        print_evaluation_methods(m, scores, x, y)
        print_feature_importance(m, x, y)


def perform_features_manipulation(df):
    numeric_features = ['OTHER_SITE_VALUE', 'NUM_DEAL', 'LAST_DEAL', 'ADVANCE_PURCHASE', 'FARE_L_Y1', 'FARE_L_Y2',
                        'FARE_L_Y3', 'FARE_L_Y4', 'FARE_L_Y5', 'POINTS_L_Y1', 'POINTS_L_Y2', 'POINTS_L_Y3',
                        'POINTS_L_Y4', 'POINTS_L_Y5']
    for col in numeric_features:
        df[col + '_bool'] = (df[col] > 0).astype(int)
        col_after_cut = pd.cut(df[col], 5, precision=0, labels=range(0, 5))
        rated_dummies = pd.get_dummies(col_after_cut,prefix=col)
        df = pd.concat([df, rated_dummies], axis=1)
        df = df.drop(col, axis=1)
    return df


if __name__ == '__main__':
    df = pd.read_csv('csv_files/hw#2/train_data/ffp_train_with_rev.csv', encoding="UTF-8")
    df = df.drop(['ID'], axis=1)
    choose_best_model(df)
    print('Try again with features manipulation')
    df = perform_features_manipulation(df)
    choose_best_model(df)




