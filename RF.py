from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split,cross_val_predict
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error as MAE
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import shap
import eli5
from eli5.sklearn import PermutationImportance
data = pd.read_excel(r'E:\nanoplasticse\RF_model\model_data.xlsx')
X = data.take([0,1,2,5,6,7,8,9,10,11],axis=1)
ohe = OneHotEncoder(categories='auto',handle_unknown='ignore').fit(X)
result = ohe.transform(X).toarray()
ohename = ohe.get_feature_names_out().tolist()
newdata = pd.concat([data,pd.DataFrame(result)],axis=1)
newdata.drop(['Core', 'Plastic source', 'Shape',
        'Surface modification', 'Cell line',
       'Cell anatomical type', 'Cell source species', 'Cell origin',
       'Cell-tissue-organ-origin', 'Assay type'
       ],axis=1,inplace=True)
numerical_columns = ['Concentration(μg/mL)','Diameter(nm)',
                'Exposure time', 'Cell viability']
for column in ohename:
    numerical_columns.append(column)
newdata.columns = numerical_columns
x=newdata.drop('Cell viability',axis=1)
y=newdata['Cell viability']  
x.columns = x.columns.astype(str)
x = StandardScaler().fit_transform(x)
x0=pd.DataFrame(x)
x0.columns = numerical_columns
# Build the model
Xtrain,Xtest,Ytrain,Ytest = train_test_split(x0,y,test_size=0.2,random_state=193)
'''

Parameter optimization and five-fold cross validation 

'''
#Random_state
score_5cv_all = []
for i in range(0, 400, 1):
    rfr =RandomForestRegressor(random_state=i)
    score_5cv =cross_val_score(rfr, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass
score_max_5cv = max(score_5cv_all)

print("Best_5cv score:{}".format(score_max_5cv),
      "random_5cv:{}".format(score_5cv_all.index(score_max_5cv)))

random_state_5cv = range(0, 400)[score_5cv_all.index(max(score_5cv_all))]
print(random_state_5cv)



#n_estimator
score_5cv_all = []
for i in range(1, 500, 1):
    rfr = RandomForestRegressor(n_estimators=i
                                , random_state=random_state_5cv)
    score_5cv = cross_val_score(rfr, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass

score_max_5cv = max(score_5cv_all)
n_est_5cv = range(1,500)[score_5cv_all.index(score_max_5cv)]
print("Best_5cv score:{}".format(score_max_5cv),
      "n_est_5cv:{}".format(n_est_5cv))



#Max_depth
score_5cv_all = []
for i in range(1, 200, 1):
    rfr = RandomForestRegressor(n_estimators=n_est_5cv
                                , random_state= random_state_5cv 
                                , max_depth=i)
    score_5cv = cross_val_score(rfr, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass
score_max_5cv = max(score_5cv_all)
print("Best_5cv score:{}".format(score_max_5cv),
      "max_depth_5cv:{}".format(score_5cv_all.index(score_max_5cv)))
max_depth_5cv = range(1,200)[score_5cv_all.index(score_max_5cv)]
print(max_depth_5cv )


#min_samples_leaf 
score_5cv_all = []
for i in range(1, 100, 1):
    rfr = RandomForestRegressor(n_estimators=n_est_5cv
                                , random_state= random_state_5cv 
                                , max_depth=max_depth_5cv
                                ,min_samples_leaf=i)
    score_5cv = cross_val_score(rfr, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass
score_max_5cv = max(score_5cv_all)
print("Best_5cv score:{}".format(score_max_5cv),
      "min_samples_leaf_5cv:{}".format(score_5cv_all.index(score_max_5cv)))
min_samples_leaf_5cv = range(1,200)[score_5cv_all.index(score_max_5cv)]
print(min_samples_leaf_5cv )

#min_samples_split
score_5cv_all = []
for i in range(1, 100, 1):
    rfr = RandomForestRegressor(n_estimators=n_est_5cv
                                , random_state= random_state_5cv 
                                , max_depth=max_depth_5cv
                                ,min_samples_leaf=min_samples_leaf_5cv
                                ,min_samples_split=i)
    score_5cv = cross_val_score(rfr, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass
score_max_5cv = max(score_5cv_all)
print("Best_5cv score:{}".format(score_max_5cv),
      "min_samples_split_5cv:{}".format(score_5cv_all.index(score_max_5cv)))
min_samples_split_5cv = range(1,200)[score_5cv_all.index(score_max_5cv)]
print(min_samples_split_5cv )




#max_features
score_5cv_all = []
for i in range(1,Xtrain.shape[1]+1):
    rfr = RandomForestRegressor(n_estimators=n_est_5cv
                                ,random_state=random_state_5cv
                                ,max_depth=max_depth_5cv
                                ,min_samples_leaf=min_samples_leaf_5cv
                                ,min_samples_split=min_samples_split_5cv
                                ,max_features=i)
    score_5cv = cross_val_score(rfr, Xtrain, Ytrain, cv=5).mean()
    score_5cv_all.append(score_5cv)
    pass

score_max_5cv = max(score_5cv_all)
max_features_5cv = range(1, Xtrain.shape[1]+1)[score_5cv_all.index(score_max_5cv)]

print("Best_5cv score:{}".format(score_max_5cv),
      "max_features_5cv:{}".format(max_features_5cv))




rfr = RandomForestRegressor(n_estimators=n_est_5cv,
                            random_state= random_state_5cv,
                            max_depth=max_depth_5cv,
                            max_features=max_features_5cv,
                            min_samples_leaf=min_samples_leaf_5cv,
                            min_samples_split = min_samples_split_5cv)
CV_score = cross_val_score(rfr, Xtrain, Ytrain, cv=5).mean()
CV_predictions = cross_val_predict(rfr, Xtrain, Ytrain, cv=5)
rmse = np.sqrt(mean_squared_error(Ytrain,CV_predictions))
mae = MAE(Ytrain, CV_predictions)
print("5cv:",CV_score)
print("rmse_5CV",rmse)
print('mae_5CV',mae)
expvspred_5cv = {'Exp': Ytrain, 'Pred':CV_predictions}
pd.DataFrame(expvspred_5cv).to_excel(r'E:\nanoplastics\nanoplastic_model\RF_model\Random_forest_5fcv_predictions.xlsx')

'''
Test set validation

'''

# rfc = RandomForestRegressor(n_estimators=n_est_5cv,random_state=random_state_5cv,max_depth=max_depth_5cv,max_features=max_features_5cv)
regressor = rfr.fit(Xtrain, Ytrain)
test_predictions = regressor.predict(Xtest)
score_test = regressor.score(Xtest,Ytest)
rmse_test = np.sqrt(mean_squared_error(Ytest,test_predictions))
mae_test = MAE(Ytest, test_predictions)
print("test:",score_test)
print("rmse_test",rmse_test)
print('mae_test', mae_test)
expvspred_test = {'Exp':Ytest,'Pred':test_predictions}
pd.DataFrame(expvspred_test).to_excel(r'E:\nanoplastics\nanoplastic_model\RF_model\RF_test_predictions.xlsx')


# Feature importance analysis
#1. random forest feature importance
feature_importance = regressor.feature_importances_
dataframe_feature_importance = sorted(zip(feature_importance, Xtrain.columns),reverse=True)
df = pd.DataFrame(dataframe_feature_importance, columns=["importance", "feature"])
df["prefix"] = df["feature"].str.split("_").str[0]
df_new = df.groupby("prefix")["importance"].sum().reset_index()
df_new.sort_values(by='importance',ascending=False,inplace=True)


# 2. permutation importance
perm = PermutationImportance(regressor, random_state=1).fit(Xtest, Ytest)
eli5.show_weights(perm,feature_names=Xtest.columns.tolist())
df = eli5.format_as_dataframe(eli5.explain_weights(perm, feature_names = Xtest.columns.tolist(), top=None))
df.to_excel(r"E:\nanoplastics\nanoplastic_model\feature_importance\feature_importance_new\permutation_importance.xlsx"
            , index=False)

# 3. SHAP analysis
shap.initjs()
explainer = shap.TreeExplainer(regressor)
shap_values = explainer.shap_values(Xtrain)
plt.subplots(figsize=(30, 15), dpi=1080, facecolor='w')
shap.summary_plot(shap_values, Xtrain, show=False, max_display=8)
plt.savefig(r"E:\nanoplastics\nanoplastic_model\shap\shap_importance_new.tif")