import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import KFold

# read data
X = pd.read_excel(r'E:\nanoplastics\nanoplastic_model\nanoplastic_std_x0.xlsx')
y = pd.read_excel(r'E:\nanoplastics\nanoplastic_model\nanoplastic_std_y.xlsx')[
    'Cell viability']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 5-fold
kf = KFold(n_splits=5, shuffle=True, random_state=0)

r2_train_list = []
r2_test_list = []
rmse_train_list = []
rmse_test_list = []
mae_train_list = []
mae_test_list = []

for train_index, test_index in kf.split(Xtrain):

    X_train, X_test = Xtrain.iloc[train_index], Xtrain.iloc[test_index]
    y_train, y_test = Ytrain.iloc[train_index], Ytrain.iloc[test_index]
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=550, batch_size=32)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    xtrain_pred = model.predict(X_train)
    mse_train = mean_squared_error(y_train, xtrain_pred)
    mae_train = mean_absolute_error(y_train, xtrain_pred)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, xtrain_pred)
    r2_train_list.append(r2_train)
    rmse_train_list.append(rmse_train)
    mae_train_list.append(mae_train)
    print('Train RMSE:', rmse_train)
    print('Train r2:', r2_train)
    print('Train RMAE:', mae_train)
    mae_train_list.append(mae_train)
mean_r2_train = np.mean(r2_train_list)
print('Average R2 train:', mean_r2_train)
mean_mae_train = np.mean(mae_train_list)
print('Average MAE train:', mean_mae_train)
mean_rmse_train = np.mean(rmse_train_list)
print('Average RMSE train:', mean_rmse_train)


# test validation

model.fit(Xtest, Ytest, epochs=550, batch_size=32)
Y_pred = model.predict(Xtest)
r2_test = r2_score(Ytest, Y_pred)
mae_test = mean_absolute_error(Ytest, Y_pred)
rmse_test = np.sqrt(mean_squared_error(Ytest, Y_pred))
print('External R2:', r2_test)
print('External MAE:', mae_test)
print('External RMSE:', rmse_test)
