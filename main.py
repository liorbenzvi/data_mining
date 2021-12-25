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
# from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier


def clean_data(df):
    df = df.drop(['ID', 'name', 'gotten', 'appli', 'ask', 'did', 'ice', 'tart', 'zico', 'etc', 'jar', 'corn', 'jam',
                  'also', 'cream', 'was', 'bbq', 'the', 'and', 'product', 'had', 'this', 'would', 'should', 'thought',
                  'for', 'box', 'tast', 'what', 'make', 'flavor', 'money', 'receiv', 'hope', 'have', 'were', 'review',
                  'are', 'too', 'has', 'gave', 'find', 'year', 'compani', 'their', 'buy', 'time', 'you', 'way', 'deal',
                  'almond', 'list', 'that', 'need', 'quit', 'again', 'been', 'wonder', 'about', 'them'], axis=1)
    return df


def get_xy(df):
    y_filter = ['rating']
    x = df[df.columns[~df.columns.isin(y_filter)]]
    y = df[y_filter].values.ravel()
    return x, y


def print_feature_importance(model):
    model.fit(x, y)
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        sorted_indices = np.argsort(importance)[::-1]
        print('20 most important features (sorted by importance): ')
        sorted_features = x.columns[sorted_indices]
        print(*sorted_features[:20], sep="\n")
    else:
        print('model does not have feature_importances_ so skipping this method')


def train_and_evaluate_model(model, x, y, cv):
    scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    print_feature_importance(model)


if __name__ == '__main__':
    df = pd.read_csv("csv_files/text_training.csv", encoding="UTF-8")
    df = clean_data(df)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    x, y = get_xy(df)
    print('going to check different models:')
    best_features = ['not', 'great', 'return', 'disappoint', 'love', 'wast', 'perfect', 'best', 'bad', 'terribl',
                    'horribl', 'threw', 'worst', 'favorit', 'disgust', 'wouldnt', 'refund', 'good', 'stale', 'noth']
    models = [('RF', RandomForestClassifier()),
              ('SVM', svm.SVC()),
              ('LR', LogisticRegression()),
              ('LDA', LinearDiscriminantAnalysis()),
              ('KNN', KNeighborsClassifier()),
              ('CART', DecisionTreeClassifier()),
              ('NB', GaussianNB()),
              # ('XGB', XGBClassifier(silent=False, n_jobs=13, random_state=15, n_estimators=100)),
              ('NN', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))]
    for name, m in models:
        print(f'\n{name}:')
        train_and_evaluate_model(m, x, y, cv)
        print('now again only with best feature')
        train_and_evaluate_model(m,
                                 df[best_features],
                                 df[['rating']].values.ravel(),
                                 cv)




