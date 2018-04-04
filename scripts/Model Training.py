import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle
import math

# Setting our classifiers
rf_clf = RandomForestClassifier()
ada_clf = AdaBoostClassifier()
bag_clf = BaggingClassifier()
grad_clf = GradientBoostingClassifier()
dt_clf = DecisionTreeClassifier()
et_clf = ExtraTreesClassifier()
xgb_clf = XGBClassifier()

# This is a list of the params I am going to test using GridSearchCV
# These are not turely optmizied features, the features that were selected
# by gridserach are included as well and some are at the limits so the limits
# need to be expended a bit
rand_dict = {'n_estimators':[300, 400, 500, 750, 1000, 1500],
            'max_features': ['auto', 'sqrt', 'log2', 0.2, 0.9],
            'min_samples_split': [3, 5, 7, 10, 15, 20],
            'criterion': ['gini', 'entropy'],
            'oob_score': [True, False]}

dt_dict = {'min_samples_split': [3, 5, 7, 10, 15, 20],
          'max_features': ['auto', 'sqrt', 'log2', 0.2, 0.9],
          'criterion': ['gini', 'entropy'],
          'splitter': ['best', 'random']}

ada_dict = {'n_estimators':[20, 40, 50, 70, 100],
           'learning_rate': [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1],
           'algorithm': ['SAMME', 'SAMME.R']}

bag_dict = {'n_estimators': [5, 7, 10, 15, 20],
           'max_samples': [0.2, 0.5, 0.9],
           'max_features': [0.2, 0.5, 0.9, 1],
           'oob_score': [True, False]}

grad_dict = {'learning_rate': [0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1],
            'n_estimators': [25, 50, 100, 300, 500, 1000],
            'loss' : ['deviance', 'exponential'],
            'criterion': ['mse', 'mae', 'friedman_mse'],
            'max_features': ['auto', 'sqrt', 'log2', 0.2, 0.9]}

xgb_dict = {'learning_rate': [0.001, 0.025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5],
           'max_depth' : [6, 7, 8, 10, 15, 20],
           'n_estimators': [750, 900, 1000, 1500, 2000],
           'booster': ['gbtree', 'dart', 'gblinear']}

et_dict = {'n_estimators': [5, 7, 10, 20, 50],
          'criterion': ['gini', 'entropy'],
          'max_features': ['auto', 'sqrt', 'log2', 0.2, 0.9],
          'min_samples_split': [3, 5, 7, 10, 15, 20]}


def train_clf(clf, features, labels, arg_dict, *name_label):
    '''
    This function trians the clf using the arg dict options passed to it.
    : param clf - This is the classifier to be trained. Is a classifier obj.
    : param features - The features to be used when fitting the grid search.
                        Is a list.
    : param labels - The labels to be used when fitting. Is a list.
    : param arg_dict - The dict of paramaters to test with gridsearch.
                        Is a dictionary.
    : arg *name_label - An optional paramater that is used when printing
    : RETURN - Returns the best trained clf and a list of scores from
                cross val score testing
    '''


    # Doing the gridsearch
    estimator = GridSearchCV(clf, param_grid=arg_dict,
                             scoring='roc_auc',
                            cv=5, n_jobs=-1)
    estimator.fit(features, labels)

    # Getting the best clf then returning the scores for it, since
    # this is a classification task using roc_auc socring
    best_clf = estimator.best_estimator_
    score = cross_val_score(best_clf, features, labels,
                            cv=10, scoring='roc_auc', n_jobs=-1)

    # Prints for making an outfile so that the best options are saved
    print('DATA FOR {}'.format(name_label))
    print('')
    print('Best CLF: {}'.format(best_clf))
    print('')

    return best_clf, score


def find_ci(list_of_scores, list_of_models):
    '''
    Find the mean and bounds of the scores.
    : param list_of_scores - A list of scores to be passed that will be used
                            in determination of the mean, std, and CI
    : param list_of_models - The models list that will be used as the index
                            for the data frame.
    : RETURNS - Data frame containing the mean, upper, lower, and CI of
                the list of scores passed to it.
    '''

    # Initialization of some variables
    mean_list = list()
    ci_list = list()

    # Loop to go through the list of scores provided to it
    for score in list_of_scores:
        mean_list.append(score.mean())
        std = score.std()
        std_error = std/math.sqrt(score.shape[0])
        ci_list.append(2.262*std_error)

    # Creation of the output df
    out_df = pd.DataFrame(index=list_of_models,
                         columns=['Mean', 'Upper', 'Lower', 'ci'])

    # Setting the values of the data frame columns
    out_df['Mean'] = mean_list
    out_df['ci'] = ci_list
    out_df['Upper'] = out_df['Mean'] + out_df['ci']
    out_df['Lower'] = out_df['Mean'] - out_df['ci']

    return out_df


# Reading in the data to be trained on
per_min = pd.read_csv('../Data/per_min_teamdf.csv', index_col=0)
per_feats = per_min.drop('result', 1)
per_labels = per_min['result'].astype('int')

# Creating a train test split so that the prediction can be done on data that
# is not tested. The random state here is 859 and will be used in creating
# the same split when predicting
per_f_train, per_f_test, per_l_train, per_l_test = train_test_split(per_feats,
    per_labels, test_size = 0.3, random_state = 859)

# Same for other data set
dropped = pd.read_csv('../Data/dropped_teamdf.csv', index_col=0)
drop_feats = dropped.drop('result', 1)
drop_labels = dropped['result'].astype('int')

drop_f_train, drop_f_test, drop_l_train, drop_l_test = train_test_split(drop_feats,
    drop_labels, test_size = 0.3, random_state=859)


# Training the models using the train_clf function described earlier
# Could be done in a loop and not line by line.
per_ada, per_ada_score = train_clf(ada_clf, per_f_train, per_l_train,
    ada_dict, 'PER')
per_bag, per_bag_score = train_clf(bag_clf, per_f_train, per_l_train,
    bag_dict, 'PER')
per_grad, per_grad_score = train_clf(grad_clf, per_f_train, per_l_train,
    grad_dict, 'PER')
per_dt, per_dt_score = train_clf(dt_clf, per_f_train, per_l_train,
    dt_dict, 'PER')
per_et, per_et_score = train_clf(et_clf, per_f_train, per_l_train,
    et_dict, 'PER')
per_rf, per_rf_score = train_clf(rf_clf, per_f_train, per_l_train,
    rand_dict, 'PER')
per_xgb, per_xgb_score = train_clf(xgb_clf, per_f_train, per_l_train,
    xgb_dict, 'PER')


# Training for other data set
drop_rf, drop_rf_score = train_clf(rf_clf, drop_f_train, drop_l_train,
    rand_dict, 'DROP')
drop_ada, drop_ada_score = train_clf(ada_clf, drop_f_train, drop_l_train,
    ada_dict, 'DROP')
drop_bag, drop_bag_score = train_clf(bag_clf, drop_f_train, drop_l_train,
    bag_dict, 'DROP')
drop_grad, drop_grad_score = train_clf(grad_clf, drop_f_train, drop_l_train,
    grad_dict, 'DROP')
drop_dt, drop_dt_score = train_clf(dt_clf, drop_f_train, drop_l_train,
    dt_dict, 'DROP')
drop_et, drop_et_score = train_clf(et_clf, drop_f_train, drop_l_train,
    et_dict, 'DROP')
drop_xgb, drop_xgb_score = train_clf(xgb_clf, drop_f_train, drop_l_train,
    xgb_dict, 'DROP')


# Creating hte score list using the find_ci function from earlier
score_list = [per_rf_score, per_dt_score, per_ada_score, per_bag_score,
             per_grad_score, per_xgb_score, per_et_score]

score_df = find_ci(score_list,
                  ['rf', 'dt', 'ada','bag', 'grad', 'xgb', 'et'])
print('Score DF: {}'.format(score_df))
print('')

# For other data set
score_list = [drop_rf_score, drop_dt_score, drop_ada_score, drop_bag_score,
             drop_grad_score, drop_xgb_score, drop_et_score]

score_df = find_ci(score_list,
                  ['rf', 'dt', 'ada', 'bag', 'grad', 'xgb', 'et'])
print('Score DF: {}'.format(score_df))
print('')


# Creating the VotingClassifier using soft voting as the sub classifiers are
# well trained to the data due to gridsearchcv.
softVoteC_drop = VotingClassifier(estimators=[('rfc', drop_rf), ('dt', drop_dt),
                                    ('ada', drop_ada), ('bag', drop_bag),
                                    ('grad', drop_grad), ('xgb', drop_xgb),
                                    ('et', drop_et)], voting='soft', n_jobs=-1)

# Repeat for other Data set
softVoteC_per = VotingClassifier(estimators=[('rfc', per_rf), ('dt', per_dt),
                                    ('ada', per_ada), ('bag', per_bag),
                                    ('grad', per_grad), ('xgb', per_xgb),
                                    ('et', per_et)], voting='soft', n_jobs=-1)

# Fitting the VoteClassifiers
softVoteC_drop = softVoteC_drop.fit(drop_f_train, drop_l_train)
softVoteC_per = softVoteC_per.fit(per_f_train, per_l_train)


# Dumping the voteClassifiers to pickle files for saving and downloading
with open('Soft_voteC_drop.pkl', 'wb') as pf:
    pickle.dump(softVoteC_drop, pf)

with open('Soft_voteC_per.pkl', 'wb') as pf:
    pickle.dump(softVoteC_per, pf)

print('Done!!!!')

