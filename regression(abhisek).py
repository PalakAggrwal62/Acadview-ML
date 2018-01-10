# Library
import pandas as pd

# Dataset
dataset = pd.read_csv("sample_data.csv")
X = dataset.iloc[:, 2:-1].values
y = dataset.iloc[:, -1].values

# Missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[:, 1:4])
X[:, 1:4] = imputer.transform(X[:, 1:4])

# Splitting Data into Training And Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Without Scalling
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

# Cheecking Accuracy
from sklearn.metrics import r2_score
acc = round(r2_score(y_test, y_pred) * 100, 2) # 87.82 when random_State = 0, default random state ranges from 67-96%

# With Scalling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

regressor_sc = LinearRegression()
regressor_sc.fit(X_train, y_train)

y_pred_sc = regressor_sc.predict(X_test)

# Checking Accuracy
acc_sc = round(r2_score(y_test, y_pred_sc) * 100, 2) # 87.82 when random_State = 0, default random state ranges from 67-96%
