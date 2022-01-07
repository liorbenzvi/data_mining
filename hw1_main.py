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


def clean_data(df):
    df = df.drop(['ID', 'name', 'gotten', 'appli', 'ask', 'did', 'ice', 'tart', 'zico', 'etc', 'jar', 'corn', 'jam',
                  'also', 'cream', 'was', 'bbq', 'the', 'and', 'product', 'had', 'this', 'would', 'should', 'thought',
                  'for', 'box', 'tast', 'what', 'make', 'flavor', 'money', 'receiv', 'hope', 'have', 'were', 'review',
                  'are', 'too', 'has', 'gave', 'find', 'year', 'compani', 'their', 'buy', 'time', 'you', 'way', 'deal',
                  'almond', 'list', 'that', 'need', 'quit', 'again', 'been', 'wonder', 'about', 'them',
                  'happen', 'idea', 'sound', 'later', 'stevia', 'guess', 'with'], axis=1)
    return df


def get_xy(df):
    y_filter = ['rating']
    x = df[df.columns[~df.columns.isin(y_filter)]]
    y = df[y_filter].values.ravel()
    return x, y


def print_feature_importance(model):
    model.fit(x, y)
    if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            importance = model.coef_
        sorted_indices = np.argsort(importance)[::-1]
        print('20 most important features (sorted by importance): ')
        sorted_features = x.columns[sorted_indices]
        if hasattr(model, 'feature_importances_'):
            print(*sorted_features[:20], sep="\n")
        else:
            print(*sorted_features[0][:20], sep="\n")
    else:
        print('model does not have feature_importances_ or coef_ so skipping this method')


def train_and_evaluate_model(model, x, y, cv):
    scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
    print_feature_importance(model)


def choose_best_model(df, cv, x, y):
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
              ('XGB', XGBClassifier(silent=False, n_jobs=13, random_state=15, n_estimators=100)),
              ('NN', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))]
    for name, m in models:
        print(f'\n{name}:')
        train_and_evaluate_model(m, x, y, cv)
        print('now again only with best feature')
        train_and_evaluate_model(m, df[best_features], df[['rating']].values.ravel(), cv)


def parameter_tuning(x, y):
    random_grid = {'n_estimators': [int(x) for x in np.linspace(start=200, stop=2000, num=10)],
                   'max_features': ['auto', 'sqrt'],
                   'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                   'min_samples_split': [2, 5, 10],
                   'min_samples_leaf': [1, 2, 4],
                   'bootstrap': [True, False]}
    rf = RandomForestClassifier()
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=100, cv=3,
                                   verbose=2, random_state=42, n_jobs=-1)
    rf_random.fit(x, y)
    print('\nBest params: ')
    print(rf_random.best_params_)


def calc_improvement(x, y):
    base_model = RandomForestClassifier()
    base_accuracy = cross_val_score(base_model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Base accuracy: %.3f (%.3f)' % (mean(base_accuracy), std(base_accuracy)))
    best_random = RandomForestClassifier(n_estimators=2000, min_samples_split=5, min_samples_leaf=1, max_features='auto'
                                         , max_depth=None, bootstrap=False)
    improved_accuracy = cross_val_score(best_random, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    print('Best accuracy: %.3f (%.3f)' % (mean(improved_accuracy), std(improved_accuracy)))
    print('Improvement of {:0.2f}%.'.format(100 * (mean(improved_accuracy) - mean(base_accuracy)) / mean(base_accuracy)))


def create_recommendation_file(x, y):
    rf = RandomForestClassifier(n_estimators=2000, min_samples_split=5, min_samples_leaf=1, max_features='auto',
                                max_depth=None, bootstrap=False)
    rf.fit(x, y)
    rollout_df = pd.read_csv("csv_files/hw#1/text_rollout_X.csv", encoding="UTF-8")
    clean_df = clean_data(rollout_df)
    clean_df = clean_df.drop("rating", axis=1)
    rollout_df['rating'] = rf.predict(clean_df)
    (rollout_df[['ID', 'rating']]).to_csv("csv_files/recommendations.csv", index=False)


if __name__ == '__main__':
    df = pd.read_csv("csv_files/hw#1/text_training.csv", encoding="UTF-8")
    df = clean_data(df)
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    x, y = get_xy(df)
    choose_best_model(df, cv, x, y)
    parameter_tuning(x, y)
    calc_improvement(x, y)
    create_recommendation_file(x,y)






