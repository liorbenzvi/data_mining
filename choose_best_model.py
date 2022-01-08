import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from sklearn import linear_model, svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from hw1_main import print_feature_importance


def get_xy(df):
    y_filter = ['BUYER_FLAG']
    x = df[df.columns[~df.columns.isin(y_filter)]]
    y = df[y_filter].values.ravel()
    return x, y


def print_evaluation_methods(model, scores, x, y):
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    model.fit(x, y)
    # todo - complete more functions


def add_rating_feature(df):
    # todo - complete this method base of hw1
    return df


def choose_best_model(df):
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    x, y = get_xy(df)
    print('going to check different models:')
    models = [('RF', RandomForestClassifier()),
              ('SVM', svm.SVC()),
              ('LR', LogisticRegression()),
              ('LDA', LinearDiscriminantAnalysis()),
              ('KNN', KNeighborsClassifier()),
              ('CART', DecisionTreeClassifier()),
              ('NB', GaussianNB()),
              ('XGB', XGBClassifier(silent=False, n_jobs=13, random_state=15, n_estimators=100)),
              ('NN', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))]
    for name, m in models:
        print(f'\nCheck model: {name}:')
        scores = cross_val_score(m, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
        print_evaluation_methods(m, scores, x, y)
        print_feature_importance(m, x, y)


if __name__ == '__main__':
    df = pd.read_csv('csv_files/hw#2/train_data/reviews_training.csv', encoding="UTF-8")
    df = add_rating_feature(df)
    choose_best_model(df)
    print('Try again with featurs manipulation')




