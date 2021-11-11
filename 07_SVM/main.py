import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

data = pd.read_csv("data/pulsar_data_train.csv").dropna(how='any')

X = pd.DataFrame(data.iloc[:, :-1]).to_numpy()
y = pd.DataFrame(data.iloc[:, -1:]).to_numpy().ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=57)

stdScaler = StandardScaler()
X_train = stdScaler.fit_transform(X_train)
X_test = stdScaler.transform(X_test)

svc = SVC(C=1.0, kernel="linear", probability=True)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
y_pred_proba = pd.DataFrame(svc.predict_proba(X_test)).iloc[:, 1]
f1 = f1_score(y_test, y_pred, average="micro")
print("F1 score:\t", f1)

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)
print("AUC score:\t", auc)
plt.plot(fpr, tpr, label="Support Vector Machine")
plt.plot([0, 1], [0, 1], label="Random classifier", linestyle="--")
plt.legend(loc=4)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve")
plt.show()
