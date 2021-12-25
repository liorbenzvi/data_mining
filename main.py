import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def clean_data(df):
    df = df.drop(['ID', 'name', 'gotten', 'appli', 'ask', 'did', 'ice', 'tart', 'zico', 'etc', 'jar', 'corn', 'jam',
                  'also', 'cream', 'was', 'bbq', 'the', 'and', 'product', 'had', 'this', 'would', 'should', 'thought',
                  'for', 'box', 'tast', 'what', 'make', 'flavor', 'money', 'receiv', 'hope', 'have', 'were', 'review'
                  ], axis=1)
    return df


def get_cvs(df):
    y_filter = ['rating']
    x = df[df.columns[~df.columns.isin(y_filter)]]
    y = df[y_filter].values.ravel()
    return KFold(n_splits=10, random_state=1, shuffle=True), x, y


def print_feature_importance(model):
    model.fit(x, y)
    importance = model.feature_importances_
    sorted_indices = np.argsort(importance)[::-1]
    print('20 most important features (sorted by importance): ')
    sorted_features = x.columns[sorted_indices]
    print(*sorted_features[:20], sep="\n")


def train_and_evaluate_model(model):
    scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    print_feature_importance(model)


if __name__ == '__main__':
    df = pd.read_csv("csv_files/text_training.csv", encoding="UTF-8")
    df = clean_data(df)
    cv, x, y = get_cvs(df)
    print('going to check different models:')
    print('\nRF:')
    train_and_evaluate_model(RandomForestClassifier())
    print('\nXGB:')
    train_and_evaluate_model(XGBClassifier(silent=False, n_jobs=13, random_state=15, n_estimators=100))


