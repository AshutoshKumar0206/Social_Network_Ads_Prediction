import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Social_Network_Ads.csv')

data['Gender'].replace({"Male": 0, "Female": 1}, inplace=True)

print(data.head()) 
x = data.iloc[:, 1:4].values
y = data.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x = scaler.fit_transform(x) #as in age and estimated salary value there is a large difference so need to scale the values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(max_depth=2, min_samples_split=2)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score
# print(model.get_params())

param_dist = {
   "criterion": ["squared_error", "friedman_mse", "absolute_error"],
   "max_depth": [1, 2, 3, 4, 5, 6, 7, None],
   "min_samples_split": [2, 5, 10],
}

from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(model, param_grid=param_dist, cv=10, n_jobs = -1)

grid.fit(x_train, y_train)

y_pred_binary = np.round(y_pred)
print('Accuracy score:', accuracy_score(y_test, y_pred_binary))

print("best estimated :", grid.best_estimator_)
print("params selected :", grid.best_params_)

print("best score :", grid.best_score_)

from sklearn.tree import plot_tree
plt.figure(figsize=(8,5))
plot_tree(model, feature_names=['Gender', 'Age', 'Salary'], filled=True)
plt.show()