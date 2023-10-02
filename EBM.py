from interpret.glassbox import ExplainableBoostingRegressor, LinearRegression, RegressionTree
from interpret import show
import pandas as pd
data = pd.read_excel(
    r'E:\nanoplastics\nanoplastic_model\nanoplastic_data.xlsx')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
seed = 1
ebm = ExplainableBoostingRegressor(random_state=seed)
ebm.fit(X, y)
ebm_global = ebm.explain_global(name='EBM')
show(ebm_global)
df = pd.DataFrame(ebm_global.data())
df.sort_values(by="scores", ascending=False, inplace=True)
df.to_excel(
    r'E:\nanoplastics\nanoplastic_model\feature_importance\ebm_importance.xlsx', index=None)
