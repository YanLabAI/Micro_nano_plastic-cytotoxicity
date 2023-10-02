
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as MAE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

data = pd.read_excel(r'E:\nanoplastics\XGB_model\model_data.xlsx')
X = data.take([0, 1, 2, 5, 6, 7, 8, 9, 10, 11],
              axis=1)

ohe = OneHotEncoder(categories='auto', handle_unknown='ignore').fit(X)
result = ohe.transform(X).toarray()
ohename = ohe.get_feature_names_out().tolist()
newdata = pd.concat([data, pd.DataFrame(result)], axis=1)
newdata.drop(['Core', 'Plastic source', 'Shape',
              'Surface modification', 'Cell line',
              'Cell anatomical type', 'Cell source species', 'Cell origin',
              'Cell-tissue-organ-origin', 'Assay type'
              ], axis=1, inplace=True)
numerical_columns = ['Concentration(μg/mL)', 'Diameter(nm)',
                     'Exposure time', 'Cell viability']
for column in ohename:
    numerical_columns.append(column)
newdata.columns = numerical_columns
x = newdata.drop('Cell viability', axis=1)
y = newdata['Cell viability']
x.columns = x.columns.astype(str)
x = StandardScaler().fit_transform(x)
x0 = pd.DataFrame(x)
x0.columns = numerical_columns
# Build the model
Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    x0, y, test_size=0.2, random_state=82)
'''

Parameter optimization and five-fold cross validation 

'''
score_5cv_all = []
for i in range(0, 12, 1):
    knn = KNeighborsRegressor(n_neighbors=i+1)
    score_5cv = cross_val_score(knn, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass

score_max_5cv = max(score_5cv_all)
n_neighbors_5cv = range(0, 12, 1)[score_5cv_all.index(max(score_5cv_all))]+1

print("Best_5cv score:{}".format(score_max_5cv),
      "n_neighbors_5cv:{}".format(n_neighbors_5cv))


score_5cv_all = []
for i in ['uniform', 'distance']:
    knn = KNeighborsRegressor(weights=i, n_neighbors=n_neighbors_5cv)
    score_5cv = cross_val_score(knn, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass

score_max_5cv = max(score_5cv_all)
weights_5cv = ['uniform', 'distance'][score_5cv_all.index(score_max_5cv)]

print("Best_5cv score:{}".format(score_max_5cv),
      "weights_5cv:{}".format(weights_5cv))

score_5cv_all = []
for i in ['brute', 'kd_tree', 'auto', 'ball_tree']:
    knn = KNeighborsRegressor(
        algorithm=i, weights=weights_5cv, n_neighbors=n_neighbors_5cv)
    score_5cv = cross_val_score(knn, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass

score_max_5cv = max(score_5cv_all)
algorithm_5cv = ['brute', 'kd_tree'][score_5cv_all.index(score_max_5cv)]

print("Best_5cv score:{}".format(score_max_5cv),
      "algorithm_5cv:{}".format(algorithm_5cv))

score_5cv_all = []
for i in range(0, 1000, 1):
    knn = KNeighborsRegressor(leaf_size=i+1, algorithm=algorithm_5cv,
                              weights=weights_5cv, n_neighbors=n_neighbors_5cv)
    score_5cv = cross_val_score(knn, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass

score_max_5cv = max(score_5cv_all)
leaf_size_5cv = range(10, 1000, 1)[score_5cv_all.index(score_max_5cv)]+1

print("Best_5cv score:{}".format(score_max_5cv),
      "leaf_size_5cv:{}".format(leaf_size_5cv))


Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    X, y, test_size=0.2, random_state=82)

knn = KNeighborsRegressor(leaf_size=leaf_size_5cv, algorithm=algorithm_5cv,
                          weights=weights_5cv, n_neighbors=n_neighbors_5cv)

CV_score = cross_val_score(knn, Xtrain, Ytrain, cv=5).mean()
CV_predictions = cross_val_predict(knn, Xtrain, Ytrain, cv=5)
rmse = np.sqrt(mean_squared_error(Ytrain, CV_predictions))
mae = MAE(Ytrain, CV_predictions)
print("r2_5cv:", CV_score, "rmse_5CV", rmse, "MAE_5CV", mae)
expvspred_5cv = {'Exp': Ytrain, 'Pred': CV_predictions}
pd.DataFrame(expvspred_5cv).to_excel('/home/chl/ML_new/KNN/KNN_5fcv_pred.xlsx')

'''
Test set validation

'''

knn = KNeighborsRegressor(leaf_size=leaf_size_5cv, algorithm=algorithm_5cv,
                          weights=weights_5cv, n_neighbors=n_neighbors_5cv)
regressor = knn.fit(Xtrain, Ytrain)
test_predictions = regressor.predict(Xtest)
score_test = regressor.score(Xtest, Ytest)
test_rmse = np.sqrt(mean_squared_error(Ytest, test_predictions))
test_mae = MAE(Ytest, test_predictions)
print("test:", score_test)
print("rmse_test", test_rmse)
print("mae_test", test_mae)
expvspred_test = {'Exp': Ytest, 'Pred': test_predictions}
pd.DataFrame(expvspred_test).to_excel(
    '/home/chl/ML_new/KNN/KNN_test_pred.xlsx')
