
from sklearn.ensemble import GradientBoostingRegressor
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
# 处理未知类别的直接忽略
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
    x0, y, test_size=0.2, random_state=179)
'''

Parameter optimization and five-fold cross validation 

'''
# Random_state
score_5cv_all = []
for i in range(1, 200, 1):
    gbdt = GradientBoostingRegressor(random_state=i)
    score_5cv = cross_val_score(gbdt, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass

score_max_5cv = max(score_5cv_all)
random_state_5cv = range(1, 200, 1)[score_5cv_all.index(max(score_5cv_all))]

print("Best_5cv score:{}".format(score_max_5cv),
      "random_5cv:{}".format(random_state_5cv))


score_5cv_all = []
for i in range(1, 300, 1):
    gbdt = GradientBoostingRegressor(
        n_estimators=i, random_state=random_state_5cv)
    score_5cv = cross_val_score(gbdt, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass

score_max_5cv = max(score_5cv_all)
n_est_5cv = range(1, 300, 1)[score_5cv_all.index(score_max_5cv)]

print("Best_5cv score:{}".format(score_max_5cv),
      "n_est_5cv:{}".format(n_est_5cv))


score_5cv_all = []
for i in range(1, 100, 1):
    gbdt = GradientBoostingRegressor(
        n_estimators=n_est_5cv, random_state=random_state_5cv, max_depth=i)
    score_5cv = cross_val_score(gbdt, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass

score_max_5cv = max(score_5cv_all)
max_depth_5cv = range(1, 100, 1)[score_5cv_all.index(score_max_5cv)]

print("Best_5cv score:{}".format(score_max_5cv),
      "max_depth_5cv:{}".format(max_depth_5cv))

gbdt = GradientBoostingRegressor(n_estimators=n_est_5cv, random_state=random_state_5cv, max_depth=max_depth_5cv
                                 )

CV_score = cross_val_score(gbdt, Xtrain, Ytrain, cv=5).mean()
CV_predictions = cross_val_predict(gbdt, Xtrain, Ytrain, cv=5)
rmse = np.sqrt(mean_squared_error(Ytrain, CV_predictions))
mae = MAE(Ytrain, CV_predictions)
print("r2_5cv:", CV_score, "rmse_5CV", rmse, "mae_5CV", mae)
expvspred_5cv = {'Exp': Ytrain, 'Pred': CV_predictions}
pd.DataFrame(expvspred_5cv).to_excel(
    '/home/chl/ML_new/GBDT/GBDT_5fcv_pred.xlsx')

'''
Test set validation

'''
regressor = gbdt.fit(Xtrain, Ytrain)
test_predictions = regressor.predict(Xtest)
test_mae = MAE(Ytest, test_predictions)
score_test = regressor.score(Xtest, Ytest)
rmse = np.sqrt(mean_squared_error(Ytest, test_predictions))
print("r2_test:", score_test, "rmse_test", rmse, "mae_test", test_mae)
expvspred_test = {'Exp': Ytest, 'Pred': test_predictions}
pd.DataFrame(expvspred_test).to_excel(
    '/home/chl/ML_new/GBDT/GBDT_test_pred.xlsx')
