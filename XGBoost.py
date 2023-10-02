
from xgboost import XGBRegressor as XGBR
from sklearn.model_selection import cross_val_score, train_test_split, cross_val_predict
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as MAE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

data = pd.read_excel(r'E:\nanoplastics\XGB_model\model_data.xlsx')
X = data.take([0, 1, 2, 5, 6, 7, 8, 9, 10, 11],axis=1)
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
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x0, y, test_size=0.2, random_state=193)
'''

Parameter optimization and five-fold cross validation 

'''
# Random_state
score_5cv_all = []

for i in range(0, 200, 1):
    xgbr = XGBR(random_state=i)
    score_5cv = cross_val_score(xgbr, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass
score_max_5cv = max(score_5cv_all)

print("Best_5cv score:{}".format(score_max_5cv),
      "random_5cv:{}".format(score_5cv_all.index(score_max_5cv)))

random_state_5cv = range(0, 200)[score_5cv_all.index(max(score_5cv_all))]
print(random_state_5cv)

# learning_rate
score_5cv_all = []
for i in np.arange(0.01, 0.5, 0.01):
    xgbr = XGBR(learning_rate=i, random_state=random_state_5cv)
    score_5cv = cross_val_score(xgbr, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass
score_max_5cv = max(score_5cv_all)
print("Best_5cv score:{}".format(score_max_5cv),
      "lr_5cv:{}".format(score_5cv_all.index(score_max_5cv)))
n_lr_5cv = np.arange(0.01, 0.5, 0.01)[score_5cv_all.index(score_max_5cv)]
print(n_lr_5cv)

# n_estimator
score_5cv_all = []
for i in range(1, 200, 1):
    xgbr = XGBR(n_estimators=i,
                learning_rate=n_lr_5cv,
                random_state=random_state_5cv)
    score_5cv = cross_val_score(xgbr, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass
score_max_5cv = max(score_5cv_all)

print("Best_5cv score:{}".format(score_max_5cv),
      "n_est_5cv:{}".format(score_5cv_all.index(score_max_5cv)))
n_est_5cv = range(1, 200)[score_5cv_all.index(score_max_5cv)]
print(n_est_5cv)

# Max_depth

score_5cv_all = []
for i in range(1, 100, 1):
    xgbr = XGBR(n_estimators=n_est_5cv,
                learning_rate=n_lr_5cv, random_state=random_state_5cv, max_depth=i)
    score_5cv = cross_val_score(xgbr, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass
score_max_5cv = max(score_5cv_all)

print("Best_5cv score:{}".format(score_max_5cv),
      "max_depth_5cv:{}".format(score_5cv_all.index(score_max_5cv)))
max_depth_5cv = range(1, 100)[score_5cv_all.index(score_max_5cv)]
print(max_depth_5cv)


# Gamma

score_5cv_all = []
for i in np.arange(0, 5, 0.05):
    xgbr = XGBR(n_estimators=n_est_5cv,
                learning_rate=n_lr_5cv, random_state=random_state_5cv, max_depth=max_depth_5cv, gamma=i)
    score_5cv = cross_val_score(xgbr, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass
score_max_5cv = max(score_5cv_all)
print("Best_5cv score:{}".format(score_max_5cv),
      "gamma_5cv:{}".format(score_5cv_all.index(score_max_5cv)))
max_gamma_5cv = np.arange(0, 5, 0.05)[score_5cv_all.index(score_max_5cv)]
print(max_gamma_5cv)


# Alpha

score_5cv_all = []
for i in np.arange(0, 5, 0.05):
    xgbr = XGBR(n_estimators=n_est_5cv,
                learning_rate=n_lr_5cv, random_state=random_state_5cv, max_depth=max_depth_5cv, gamma=max_gamma_5cv, alpha=i)
    score_5cv = cross_val_score(xgbr, Xtrain, Ytrain, cv=5).mean()
    CV_predictions = cross_val_predict(xgbr, Xtrain, Ytrain, cv=5)
    score_5cv_all.append(score_5cv)
    pass
score_max_5cv = max(score_5cv_all)
print("Best_5cv score:{}".format(score_max_5cv),
      "alpha_5cv:{}".format(score_5cv_all.index(score_max_5cv)))
max_alpha_5cv = np.arange(0, 5, 0.05)[score_5cv_all.index(score_max_5cv)]
print(max_alpha_5cv)

xgbr = XGBR(learning_rate=n_lr_5cv, n_estimators=n_est_5cv, random_state=random_state_5cv,
            max_depth=max_depth_5cv, gamma=max_gamma_5cv, alpha=max_alpha_5cv)
CV_score = cross_val_score(xgbr, Xtrain, Ytrain, cv=5).mean()
CV_predictions = cross_val_predict(xgbr, Xtrain, Ytrain, cv=5)
rmse = np.sqrt(mean_squared_error(Ytrain, CV_predictions))
mae = MAE(Ytrain, CV_predictions)
print("CV_MAE:", mae)
print("5cv:", CV_score)
print("RMSE_5CV", rmse)
expvspred_5cv = {'Exp': Ytrain, 'Pred': CV_predictions}
pd.DataFrame(expvspred_5cv).to_excel(r'E:\nanoplastics\nanoplastic_model\XGB_model_new\XGBoost_5fcv_predictions_test.xlsx')


'''
Test set validation

'''

XGB = XGBR(learning_rate=n_lr_5cv, n_estimators=n_est_5cv, random_state=random_state_5cv,
           max_depth=max_depth_5cv, gamma=max_gamma_5cv, alpha=max_alpha_5cv)
regressor = XGB.fit(Xtrain, Ytrain)
test_predictions = regressor.predict(Xtest)
score_test = regressor.score(Xtest, Ytest)
rmse = np.sqrt(mean_squared_error(Ytest, test_predictions))
mae_test = MAE(Ytest, test_predictions)
print("test_MAE:", mae_test)
print("test:", score_test)
print("rmse_test", rmse)
expvspred_test = {'Exp': Ytest, 'Pred': test_predictions}
pd.DataFrame(expvspred_test).to_excel(r'E:\nanoplastics\nanoplastic_model\XGB_model_new\XGB_test_predictions.xlsx')
