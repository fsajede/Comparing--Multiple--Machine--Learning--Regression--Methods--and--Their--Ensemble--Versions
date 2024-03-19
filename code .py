# import libraries
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import VotingRegressor
from scipy.stats import ttest_rel


# import dataset
lis = datasets.fetch_openml(data_id=42225)

lis
lis.feature_names
lis.data

# lis.target
lis.data.info()
lis_Table = pd.DataFrame(data=lis.data, columns=lis.feature_names)
lis_Table['target'] = lis.target
lis_Table

# checking the null values
lis_Table.isnull().sum()

# Transforming Nominal Features
ct = ColumnTransformer([("encoder", OneHotEncoder(sparse=False), [1,2,3])], remainder="passthrough")
new_data = ct.fit_transform(lis.data)
print(new_data)
#print(type(new_data))
ct.get_feature_names_out()

lis_new_data = pd.DataFrame(new_data, columns = ct.get_feature_names_out(), index = lis.data.index)
lis_new_data


lis.target
lis_new_data.info()
lis_Table_new = pd.DataFrame(data=lis_new_data, columns=ct.get_feature_names_out())
lis_Table_new['target'] = lis.target
lis_Table_new

# Linear regression
lr = LinearRegression()
scores = cross_validate(lr, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
scores
df_scores = pd.DataFrame(scores)
df_scores
rmse_lr = 0-scores["test_score"]
rmse_Linear =rmse_lr.mean()
print(rmse_Linear)

# Decision trees 
dtr = DecisionTreeRegressor()
parameters = [{"min_samples_leaf":[2,4,6,8,10]}]
tuned_dtc = GridSearchCV(dtr, parameters, scoring="neg_root_mean_squared_error", cv=5)
cv = cross_validate(tuned_dtc, lis_new_data, lis.target, scoring="neg_root_mean_squared_error", cv=10,return_train_score=True)
rmse_dtr = 0 - cv["test_score"]
rmse_tree = rmse_dtr.mean()
print(rmse_tree)

# K-nearest neighbor 
neigh = KNeighborsRegressor()
parameters_k = [{"n_neighbors":[3,5,7,9,11,13,15]}]
tuned_knn = GridSearchCV(KNeighborsRegressor(), parameters_k, scoring="neg_root_mean_squared_error", cv=5)
scores = cross_validate(tuned_knn, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
rmse_knn = 0 - scores["test_score"]
rmse_neighbor =rmse_knn.mean()
print(rmse_neighbor)

# Support-vector machines
svr = SVR()
scores = cross_validate(svr , lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
rmse_svr = 0 - scores["test_score"]
rmse_support = rmse_svr.mean()
print(rmse_support)


# Bagged Regressors
# Bagged Regressors(Linear regression)
bagged_lr = BaggingRegressor(base_estimator=LinearRegression())
scores = cross_validate(bagged_lr, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
rmse_bagged_lr = 0 - scores["test_score"]
rmse_bagged_Linear = rmse_bagged_lr.mean()
print(rmse_bagged_Linear)

# Bagged Regressors(Decision trees)
bagged_dtr = BaggingRegressor( base_estimator = tuned_dtc)
scores = cross_validate(bagged_dtr , lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
rmse_bagged_dtr = 0 - scores["test_score"]
rmse_bagged_tree =rmse_bagged_dtr.mean()
print(rmse_bagged_tree)

# Bagged Regressors(K-nearest neighbor)
bagged_neigh = BaggingRegressor( base_estimator = tuned_knn)
scores = cross_validate(bagged_neigh , lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
rmse_bagged_neigh = 0 - scores["test_score"]
rmse_bagged_neighbor = rmse_bagged_neigh.mean()
print(rmse_bagged_neighbor)

# Bagged Regressors(Support-vector machines)
bagged_svr = BaggingRegressor(base_estimator=SVR(n_estimators=2))
scores = cross_validate(bagged_svr, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
rmse_bagged_svr = 0 - scores["test_score"]
rmse_bagged_support = rmse_bagged_svr.mean()
print(rmse_bagged_support)

# Boosted Regressors(Linear regression)
boosted_lr = AdaBoostRegressor(base_estimator=LinearRegression())
scores = cross_validate(boosted_lr, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error") 
rmse_boosted_lr = 0 - scores["test_score"]
rmse_boosted_Linear = rmse_boosted_lr.mean()
print(rmse_boosted_Linear)

# Boosted Regressors(Decision trees)
boosted_dtr = AdaBoostRegressor( base_estimator = tuned_dtc)
scores = cross_validate(boosted_dtr, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error") 
rmse_boosted_dtr = 0 - scores["test_score"]
rmse_boosted_tree =rmse_boosted_dtr.mean()
print(rmse_boosted_tree)

# Boosted Regressors(K-nearest neighbor)
boosted_neigh = AdaBoostRegressor( base_estimator = tuned_knn)
scores = cross_validate(boosted_neigh, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error") 
rmse_boosted_neigh = 0 - scores["test_score"]
rmse_boosted_neighbor = rmse_boosted_neigh.mean()
print(rmse_boosted_neighbor)

# Boosted Regressors(Support-vector machines)
boosted_svr = AdaBoostRegressor(base_estimator=SVR(n_estimators=2))
scores = cross_validate(boosted_svr, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
rmse_boosted_svr = 0 - scores["test_score"]
rmse_boosted_support = rmse_boosted_svr.mean()
print(rmse_boosted_support)

# Voting regressor (heterogeneous ensemble )
vr = VotingRegressor([("lr", LinearRegression()), ("dtr", tuned_dtc),("neigh",tuned_knn),("svr",SVR())])
scores_voting = cross_validate(vr, lis_new_data, lis.target, cv=10, scoring="neg_root_mean_squared_error")
rmse_voting = 0 - scores_voting["test_score"]
rmse_vot = rmse_voting.mean()
print(rmse_vot)

# Statistical Significance
p_value_linear_and_bagged =ttest_rel(rmse_lr, rmse_bagged_lr)[1]
print(p_value_linear_and_bagged)

p_value_tree_and_bagged =ttest_rel(rmse_dtr, rmse_bagged_dtr)[1]
print(p_value_tree_and_bagged)

p_value_neighbor_and_bagged =ttest_rel(rmse_knn, rmse_bagged_neigh)[1]
print(p_value_neighbor_and_bagged)

p_value_support_and_bagged =ttest_rel(rmse_support,rmse_bagged_support)[1]
print(p_value_support_and_bagged)

print('__________________________')

p_value_linear_and_boosted =ttest_rel(rmse_lr, rmse_boosted_lr)[1]
print(p_value_linear_and_boosted)

p_value_tree_and_boosted =ttest_rel(rmse_dtr, rmse_boosted_dtr)[1]
print(p_value_tree_and_boosted)

p_value_neighbor_and_boosted =ttest_rel(rmse_knn, rmse_boosted_neigh)[1]
print(p_value_neighbor_and_boosted)

p_value_support_and_boosted =ttest_rel(rmse_svr,rmse_boosted_support)[1]
print(p_value_support_and_boosted)

print('__________________________')

p_value_linear_and_voting = ttest_rel(rmse_lr, rmse_voting)[1]
print(p_value_linear_and_voting)

p_value_tree_and_voting = ttest_rel(rmse_dtr, rmse_voting)[1]
print(p_value_tree_and_voting)

p_value_neighbor_and_voting =ttest_rel(rmse_knn, rmse_voting)[1]
print(p_value_neighbor_and_voting)

p_value_support_and_voting =ttest_rel(rmse_svr, rmse_voting)[1]
print(p_value_support_and_voting)

print('__________________________')

p_value_linear_and_tree = ttest_rel(rmse_dtr, rmse_lr)[1]
print(p_value_linear_and_tree)

p_value_linear_and_neighbor = ttest_rel(rmse_dtr, rmse_knn)[1]
print(p_value_linear_and_neighbor)

p_value_linear_and_support = ttest_rel(rmse_dtr, rmse_svr)[1]
print(p_value_linear_and_support)


#Comparing the base method and its bagged version
data = {'Linear regression':[rmse_Linear,rmse_bagged_Linear,p_value_linear_and_bagged],  'Decision trees':[rmse_tree,rmse_bagged_tree,p_value_tree_and_bagged] ,
        'K-nearest neighbor':[rmse_neighbor,rmse_bagged_neighbor,p_value_neighbor_and_bagged],"Support vector machine":[rmse_support,rmse_bagged_support,p_value_support_and_bagged]}
df = pd.DataFrame(data,index =['Base','Bagged','p_value'])
df

#Comparing the base method and its boosted version
data = {'Linear regression':[rmse_Linear,rmse_boosted_Linear,p_value_linear_and_boosted],  'Decision trees':[rmse_tree,rmse_boosted_tree,p_value_tree_and_boosted] ,
        'K-nearest neighbor':[rmse_neighbor,rmse_boosted_neighbor,p_value_neighbor_and_boosted],"Support vector machine":[rmse_support,rmse_boosted_support,p_value_support_and_boosted]}
df = pd.DataFrame(data,index =['Base','Boosted','p_value'])
df

#Comparing the base method and voting
data = {'Linear regression':[rmse_Linear,rmse_vot ,p_value_linear_and_voting],  'Decision trees':[rmse_tree,rmse_vot ,p_value_tree_and_voting] ,
        'K-nearest neighbor':[rmse_neighbor,rmse_vot ,p_value_neighbor_and_voting],"Support vector machine":[rmse_support,rmse_vot ,p_value_support_and_voting]}
df = pd.DataFrame(data,index =['Base','Voting','p_value'])
df