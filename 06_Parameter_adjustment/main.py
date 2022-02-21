# Llibreries que necessitarem
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

random_value = 33
# Carrega de dades i preparació de les dades emprant Pandas
data = pd.read_csv("data/day.csv")
datos = pd.DataFrame(data.iloc[:, 4:13])  # Seleccionam totes les files i les columnes per index
valors = pd.DataFrame(data.iloc[:, -1])  # Seleccionam totes les files i la columna objectiu

X = datos.to_numpy()
y = valors.to_numpy().ravel()
features_names = datos.columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_value)

random_forest = RandomForestRegressor(random_state=random_value)
parameters = {'max_depth': range(1, 302, 50), 'n_estimators': range(1, 302, 50)}
grid_search_cv = GridSearchCV(random_forest, parameters, cv=3)
grid_search_cv.fit(X_train, y_train)
pd.set_option('display.max_columns', None)
print(pd.DataFrame(grid_search_cv.cv_results_))
print("Best estimator: ", grid_search_cv.best_estimator_)

best_random_forest = grid_search_cv.best_estimator_
best_random_forest.fit(X_train, y_train)
y_pred = best_random_forest.predict(X_test)
print("R2 score", r2_score(y_test, y_pred))
