import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

data = pd.read_csv("processed_restaurant_clf.csv")
data = data.drop(columns=['Unnamed: 0'], errors="ignore")
data_numeric = data.select_dtypes(include=["number"])

# print(data["rate"].value_counts())
# we can see that the classes are balanced, so we will use metrics ROC-AUC, Accuracy, F1

# data splitting train/dev/test

X = data.drop(["rate"], axis=1)
y = data["rate"]


# the strategy is to find the best model using only the numeric features and then try to better
# the model with encoding of category features

X_numeric = data_numeric.drop(["rate"], axis=1)
y_numeric = data_numeric["rate"]

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_dev, X_test, y_dev, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

X_numeric_train, X_numeric_temp, y_numeric_train, y_numeric_temp = train_test_split(X_numeric, y_numeric, test_size=0.3, random_state=42, stratify=y_numeric)
X_numeric_dev, X_numeric_test, y_numeric_dev, y_numeric_test = train_test_split(X_numeric_temp, y_numeric_temp, test_size=0.5, random_state=42, stratify=y_numeric_temp)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_numeric_train)
X_dev_scaled = scaler.transform(X_numeric_dev)
X_test_scaled = scaler.transform(X_numeric_test)

# baseline

# y_pred_baseline = np.zeros_like(y_test)

# logistic regression

# model = LogisticRegression(max_iter=1000)
# model.fit(X_numeric_train, y_numeric_train)
#
# y_dev_pred = model.predict(X_numeric_dev)
# y_dev_prob = model.predict_proba(X_numeric_dev)[:, 1]
#
# y_test_pred = model.predict(X_numeric_test)
# y_test_proba = model.predict_proba(X_numeric_test)[:, 1]
#
# k-nearest neighbors
#
# knn = KNeighborsClassifier(n_neighbors=8)
# knn.fit(X_train_scaled, y_train)
#
# y_dev_pred = knn.predict(X_dev_scaled)
# y_dev_proba = knn.predict_proba(X_dev_scaled)[:, 1]
#
# was evaluated best n to be 3
#
# best_knn = KNeighborsClassifier(n_neighbors=3)
# best_knn.fit(X_train_scaled, y_train)
#
# y_test_pred = best_knn.predict(X_test_scaled)
# y_test_proba = best_knn.predict_proba(X_test_scaled)[:, 1]
#
# decision tree
#
# best_score = 0
# best_params = {}
#
# for depth in [3, 5, 7, 9]:
#     for min_split in [2, 5, 10, 20]:
#         for min_leaf in [1, 3, 5, 7, 9]:
#             dt = DecisionTreeClassifier(max_depth=depth, min_samples_split=min_split, min_samples_leaf=min_leaf, random_state=42)
#             dt.fit(X_numeric_train, y_numeric_train)
#
#             y_dev_pred = dt.predict(X_numeric_dev)
#             score = f1_score(y_numeric_dev, y_dev_pred)
#
#             if score > best_score:
#                 best_score = score
#                 best_params = {"max_depth": depth, "min_samples_split": min_split, "min_samples_leaf": min_leaf}
#
# print(best_params)
# code above evaluates best parameters for the dt
#
# best_dt = DecisionTreeClassifier(max_depth=9, min_samples_split=2, min_samples_leaf=1, random_state=42)
# best_dt.fit(X_numeric_train, y_numeric_train)
#
# y_test_pred = best_dt.predict(X_numeric_test)
# y_test_proba = best_dt.predict_proba(X_numeric_test)[:, 1]
#
# # random forest
#
# best_score = 0
# best_params = {}
#
# for n in [50, 100, 150, 200, 250, 300]:
#     for depth in [None, 5, 10, 15, 20]:
#
#         rf = RandomForestClassifier(n_estimators=n, max_depth=depth,  random_state=42)
#         rf.fit(X_numeric_train, y_numeric_train)
#
#         y_dev_pred = rf.predict(X_numeric_dev)
#
#         score = f1_score(y_numeric_dev, y_dev_pred)
#
#         if score > best_score:
#             best_score = score
#             best_params = {"n_estimators": n, "max_depth": depth}
#
# print(best_params)
#
# best_rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
#
# best_rf.fit(X_numeric_train, y_numeric_train)
#
# y_test_pred = best_rf.predict(X_numeric_test)
# y_test_proba = best_rf.predict_proba(X_numeric_test)[:, 1]
#
# # adaboost
#
# best_score = 0
# best_params = {}
#
# for n in [50, 100, 150, 200, 250, 300]:
#     for lr in [0.001, 0.01, 0.1, 1.0]:
#         ada = AdaBoostClassifier(n_estimators=n, learning_rate=lr)
#         ada.fit(X_numeric_train, y_numeric_train)
#
#         y_dev_pred = ada.predict(X_numeric_dev)
#
#         score = f1_score(y_numeric_dev, y_dev_pred)
#
#         if score > best_score:
#             best_score = score
#             best_params = {"n_estimators": n, "lr": lr}
#
# print(best_params)
#
# best_ada = AdaBoostClassifier(n_estimators=50, learning_rate=0.001, random_state=42)
# best_ada.fit(X_numeric_train, y_numeric_train)
#
# y_test_pred = best_ada.predict(X_numeric_test)
# y_test_proba = best_ada.predict_proba(X_numeric_test)[:, 1]

# svm

# best_score = 0
# best_params = {}
#
# for ker in ["linear", "poly", "rbf", "sigmoid"]:
#      model = SVC(kernel=ker, random_state=42)
#      model.fit(X_train_scaled, y_train)
#
#      y_dev_pred = model.predict(X_dev_scaled)
#      score = f1_score(y_dev, y_dev_pred)
#
#      if score > best_score:
#          best_score = score
#          best_params = {"kernel": ker}
#
# print(best_params)

# best_svm = SVC(kernel="rbf", probability=True, random_state=42)
# best_svm.fit(X_train_scaled, y_train)
# y_pred = best_svm.predict(X_test_scaled)
# y_pred_proba = best_svm.predict_proba(X_test_scaled)[:, 1]

# random forest with encoded non-numeric features

X_train_enc = pd.get_dummies(X_train, drop_first=True)
X_dev_enc = pd.get_dummies(X_dev, drop_first=True)
X_test_enc = pd.get_dummies(X_test, drop_first=True)

X_train_enc, X_dev_enc = X_train_enc.align(X_dev_enc, join="left", axis=1, fill_value=0)
X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join="left", axis=1, fill_value=0)

best_rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42)
best_rf.fit(X_train_enc, y_train)

y_dev_pred = best_rf.predict(X_dev_enc)
y_dev_proba = best_rf.predict_proba(X_dev_enc)[:, 1]

y_test_pred = best_rf.predict(X_test_enc)
y_test_proba = best_rf.predict_proba(X_test_enc)[:, 1]

# plotting

y_true = y_test
y_pred = y_test_pred

# small noise so points don't overlap

cm = confusion_matrix(y_numeric_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix - Random Forest")
plt.show()




