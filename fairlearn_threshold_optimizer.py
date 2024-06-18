import json

import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sklearn.neural_network

data = pd.read_csv("drug_data.csv")
data["Coke"] = pd.Series(data["Coke"], dtype=pd.StringDtype())
#
data["Gender"].replace(to_replace=0.48246, value=1.0, inplace=True, regex=True)
data["Gender"].replace(to_replace=-0.48246, value=0.0, inplace=True, regex=True)
data["Coke"].replace(to_replace='CL0', value='0.0', inplace=True, regex=True)
data["Coke"].replace(to_replace='CL1', value='0.0', inplace=True, regex=True)
data["Coke"].replace(to_replace='CL2', value='1.0', inplace=True, regex=True)
data["Coke"].replace(to_replace='CL3', value='1.0', inplace=True, regex=True)
data["Coke"].replace(to_replace='CL4', value='1.0', inplace=True, regex=True)
data["Coke"].replace(to_replace='CL5', value='1.0', inplace=True, regex=True)
data["Coke"].replace(to_replace='CL6', value='1.0', inplace=True, regex=True)

# print(data["Coke"])
data.to_csv('drug_data.csv', encoding='utf-8', index=False)

# data_1 = pd.read_csv('drug_data.csv')
# for i in range(len(data_1)):
#     if data_1['Gender'][i] == 1:
#         data_1.drop(i, inplace=True)
#
# data_2 = pd.read_csv('drug_data.csv')
# for i in range(len(data_2)):
#     if data_2['Gender'][i] == 0:
#         data_2.drop(i, inplace=True)

X_1 = data.drop(
    ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin',
     'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA', 'ID'], axis=1)
y_1 = data['Coke']

a_1 = data['Gender']


X_1_train, X_1_temp, y_1_train, y_1_temp, a_1_train, a_1_temp = train_test_split(X_1, y_1, a_1, test_size=0.4, random_state=42)
X_1_test, X_1_val, y_1_test, y_1_val, a_1_test, a_1_val = train_test_split(X_1_temp, y_1_temp, a_1_temp, test_size=0.5, random_state=42)

X_1_train = np.array(X_1_train, dtype=float)
y_1_train = np.array(y_1_train, dtype=float)
X_1_test = np.array(X_1_test, dtype=float)
y_1_test = np.array(y_1_test, dtype=float)
X_1_val = np.array(X_1_val, dtype=float)
y_1_val = np.array(y_1_val, dtype=float)
a_1_train = np.array(a_1_train, dtype=float)
a_1_test = np.array(a_1_test, dtype=float)
a_1_val = np.array(a_1_val, dtype=float)

batch_size = 128

MLP = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=11, max_iter=4000, batch_size=batch_size, solver='adam')

MLP.fit(X_1_train, y_1_train)
y_pred = MLP.predict(X_1_test)
f_pred = []
m_pred = []
f_true = []
m_true = []

for i in range(len(X_1_test)):
    if X_1_test[i][1] == 1:
        f_true.append(y_1_test[i])
        f_pred.append(y_pred[i])
    else:
        m_true.append(y_1_test[i])
        m_pred.append(y_pred[i])

female_CM = confusion_matrix(f_true, f_pred)
male_CM = confusion_matrix(m_true, m_pred)
print("Female CM\n", female_CM)
print("Male CM\n", male_CM)



female_dp = (female_CM[1][1] + female_CM[0][1]) / (
                female_CM[0][0] + female_CM[0][1] + female_CM[1][0] + female_CM[1][1])
male_dp = (male_CM[1][1] + male_CM[0][1]) / (male_CM[0][0] + male_CM[0][1] + male_CM[1][0] + male_CM[1][1])
female_TPR = female_CM[1][1] / (female_CM[1][1] + female_CM[1][0])
male_TPR = male_CM[1][1] / (male_CM[1][1] + male_CM[1][0])
female_FPR = female_CM[0][1] / (female_CM[0][1] + female_CM[0][0])
male_FPR = male_CM[0][1] / (male_CM[0][1] + male_CM[0][0])
print('Female TPR', female_TPR)
print('male TPR', male_TPR)
print('DP', abs(female_dp - male_dp))
print('EOP', abs(female_TPR - male_TPR))
print('EoD', 0.5 * (abs(female_FPR - male_FPR) + abs(female_TPR - male_TPR)))
print("Accuracy score on test set: ", sklearn.metrics.accuracy_score(y_1_test, y_pred))


threshold_optimizer = ThresholdOptimizer(
    estimator=MLP,
    constraints="equalized_odds",
    objective="accuracy_score",
    prefit=True,
    predict_method="auto"
    )
threshold_optimizer.fit(X_1_train, y_1_train, sensitive_features=a_1_train)
y_pred = threshold_optimizer.predict(X_1_test, sensitive_features=a_1_test)
threshold_rules_by_group = threshold_optimizer.interpolated_thresholder_.interpolation_dict
# print(json.dumps(threshold_rules_by_group, default=str, indent=4))
# plot_threshold_optimizer(threshold_optimizer)
f_pred = []
m_pred = []
f_true = []
m_true = []

for i in range(len(X_1_test)):
    if X_1_test[i][1] == 1:
        f_true.append(y_1_test[i])
        f_pred.append(y_pred[i])
    else:
        m_true.append(y_1_test[i])
        m_pred.append(y_pred[i])

female_CM = confusion_matrix(f_true, f_pred)
male_CM = confusion_matrix(m_true, m_pred)
print("Female CM\n", female_CM)
print("Male CM\n", male_CM)



female_dp = (female_CM[1][1] + female_CM[0][1]) / (
                female_CM[0][0] + female_CM[0][1] + female_CM[1][0] + female_CM[1][1])
male_dp = (male_CM[1][1] + male_CM[0][1]) / (male_CM[0][0] + male_CM[0][1] + male_CM[1][0] + male_CM[1][1])
female_TPR = female_CM[1][1] / (female_CM[1][1] + female_CM[1][0])
male_TPR = male_CM[1][1] / (male_CM[1][1] + male_CM[1][0])
female_FPR = female_CM[0][1] / (female_CM[0][1] + female_CM[0][0])
male_FPR = male_CM[0][1] / (male_CM[0][1] + male_CM[0][0])
print('Female TPR', female_TPR)
print('male TPR', male_TPR)
print('DP', abs(female_dp - male_dp))
print('EOP', abs(female_TPR - male_TPR))
print('EoD', 0.5 * (abs(female_FPR - male_FPR) + abs(female_TPR - male_TPR)))
print("Accuracy score on test set: ", sklearn.metrics.accuracy_score(y_1_test, y_pred))
print(confusion_matrix(y_1_test, y_pred))