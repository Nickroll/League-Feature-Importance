
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix, auc
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Setting some matplotlib params
rcParams.update({'figure.autolayout': True})
plt.style.use('fivethirtyeight')

# Since XGClassifier returns a .feature_importances_ attribute but sets
# all the values to NaN when used with voteClassifier and then pickled
# I am dropping that value for the feature importance calculation
from xgboost import XGBClassifier
xgbc = XGBClassifier()

# Reading in the data as a data frame
per_df = pd.read_csv('../Data/per_min_teamdf.csv', index_col=0)
dropped_df = pd.read_csv('../Data/dropped_teamdf.csv', index_col=0)

# Setting up features and labels. Trying to predict result
per_feats = per_df.drop('result', 1)
per_label = per_df['result'].astype('int')

drop_feats = dropped_df.drop('result', 1)
drop_label = dropped_df['result'].astype('int')

# Using train test split to generate the same split as was done in the
# maching learning. In this case random_state=859 was used in both
per_f_train, per_f_test, per_l_train, per_l_test = train_test_split(per_feats,
    per_label, test_size=0.3, random_state=859)
drop_f_train, drop_f_test, drop_l_train, drop_l_test = train_test_split(drop_feats,
    drop_label, test_size=0.3, random_state=859)

# Opening the voteclassifiers that were created using AWS EC2
with open('../output/Soft_voteC_per.pkl', 'rb') as pf:
    per_voteC = pickle.load(pf)

with open('../output/Soft_voteC_drop.pkl', 'rb') as pf:
    drop_voteC = pickle.load(pf)

# Setting up the prediction data frames.
per_predictions_df = pd.DataFrame(columns=['outcome', 'probability'])
drop_predictions_df = pd.DataFrame(columns=['outcome', 'probability'])

# Doing the prediction on the test samples using the vote classifier,
# also getting the probability for each prediction (0 or 1)
per_predict = per_voteC.predict(per_f_test)
per_probs = per_voteC.predict_proba(per_f_test)

# Adding the predictions to the data frames made earlier
# The np.max list comprehension just returns the max probability, i.e.
# the probability that corresponds to the prediction
per_predictions_df['outcome'] = per_predict
per_predictions_df['probability'] = [np.max(per_probs[i]) for i in range(len(per_probs))]

drop_predict = drop_voteC.predict(drop_f_test)
drop_probs = drop_voteC.predict_proba(drop_f_test)

drop_predictions_df['outcome'] = drop_predict
drop_predictions_df['probability'] = [np.max(drop_probs[i]) for i in range(len(drop_probs))]



# Getting the accuracy score and the confusion matrix for the predictions

per_acc = accuracy_score(per_l_test, per_predict)
per_cm = confusion_matrix(per_l_test, per_predict)
print('Accuracy per: {:.2%}'.format(per_acc))

# Creating the heatmap of the confusion matrix
sns.heatmap(per_cm, annot=True, fmt='.0f')
plt.title('Per Min Accuracy')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Getting the roc_auc score which is a better estimate of model accuracy
# than accuracy is for binary classifiers
per_roc_auc_score = roc_auc_score(per_l_test, per_predict)
print('Roc auc score per: {:.2%}'.format(per_roc_auc_score))

fpr, tpr, _ = roc_curve(per_l_test, per_predict)
per_auc = auc(fpr, tpr)

# Making the roc_auc graph
plt.plot(fpr, tpr, label = 'ROC curve {:.3}'.format(per_auc))
plt.plot([0,1],[0,1], linestyle = '--')
plt.title('Per Min ROC AUC')
plt.legend(loc='lower right')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.show()


feat_list_per = per_f_test.columns.values


# Function to find the sum of the feature importances across all the
# models that are present in the vote classifier.
def find_importances(voteC, feat_list, *classtype):
    '''
    Function to find and add all the feature importances together

    : param voteC - The vote classifier to be iterated through. Is a
                    vote classifier object from sklearn
    : param feat_list - The features list to be used as keys in a dict
                        when the feature importances are added. Is of
                        list type.
    : *arg classtype - Optional argument for classifiers that return
                        a .feature_importances_ attribute but set it to
                        NaN as like XGBClassifier. Is a tuple.
    '''
    feat_dict = dict()

    # Loop to iterate through all the esimators in voteC param
    for f,_ in enumerate(voteC.estimators_):
        # If the estimator does not have a .feature_importances_ attribute
        # than it will continue on to the next due to the try except
        try:

            # Loop to get the 'real' type of the options in the tuple
            # passed to *classtype
            for ct,_ in enumerate(classtype):
                # If the type of the estimator is NOT the same as the type
                # passed in one of the classtype tuples then this will
                # create a dict of the feature : feature_importances_
                if not isinstance(voteC.estimators_[f], type(classtype[ct])):
                    to_add = dict(zip(feat_list,
                        voteC.estimators_[f].feature_importances_))

                    # Loop to get the key and value pairs from the to_add
                    # dict created before and add them to the feat_dict
                    # if the key is present already in the feat_dict, if
                    # the key is not present then feat_dict is initialized
                    # as to_add.
                    for k,v in to_add.items():
                        if k in feat_dict:
                            feat_dict[k] = feat_dict[k] + v
                        else:
                            feat_dict = to_add
        except:
            continue

    # Retuns the dict of {feature: feature_importance sum}
    return feat_dict

# Getting the feature importances
per_feat_ranking = find_importances(per_voteC,
                                   feat_list_per,
                                   xgbc)

# Sorting the feature importances then grabbing the top 5
sort_per = sorted(per_feat_ranking.items(), key=lambda x:x[1],
                  reverse=True)[:5]

x = [x[0] for x in sort_per]
y = [y[1] for y in sort_per]

sns.barplot(x, y)
plt.xlabel('Feature')
plt.ylabel('Combined Importance')
plt.title('Per Min Added Feature Importance Across \nvoteC Models Excludes XGBC and Bagging')
plt.show()

# Repeat of the above for the dropped feature models
drop_acc = accuracy_score(drop_l_test, drop_predict)
drop_cm = confusion_matrix(drop_l_test, drop_predict)
print('Accuracy drop: {:.2%}'.format(drop_acc))

sns.heatmap(drop_cm, annot=True, fmt='.0f')
plt.title('Dropped Accuracy')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


drop_roc_auc_score = roc_auc_score(drop_l_test, drop_predict)
print('Roc auc score dropped: {:.2%}'.format(drop_roc_auc_score))

fpr, tpr, _ = roc_curve(drop_l_test, drop_predict)
drop_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label = 'ROC curve {:.3}'.format(drop_auc))
plt.plot([0,1],[0,1], linestyle = '--')
plt.title('Dropped ROC AUC')
plt.legend(loc='lower right')
plt.xlabel('False Positive')
plt.ylabel('True Positive')
plt.show()


feat_list_drop = drop_f_test.columns.values
drop_feat_ranking = find_importances(drop_voteC,
                                   feat_list_drop,
                                   xgbc)

sort_drop = sorted(drop_feat_ranking.items(), key=lambda x:x[1],
                  reverse=True)[:5]

x = [x[0] for x in sort_drop]
y = [y[1] for y in sort_drop]


sns.barplot(x, y)
plt.xlabel('Feature')
plt.ylabel('Combined Importance')
plt.title('Dropped Added Feature Importance Across \nvoteC Models Excludes XGBC and Bagging')
plt.show()

