import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sklearn.neural_network

from FullyConnectedNet import FullyConnectedNet

data1 = pd.read_csv("Members_Y1.csv")
data2 = pd.read_csv("Claims_Y1.csv")
data3 = pd.read_csv("DayInHospital_Y2.csv")

# data2["specialty"].replace(to_replace='Anesthesiology', value=1, inplace=True, regex=True)
# data2["specialty"].replace(to_replace='Diagnostic Imaging', value=2, inplace=True, regex=True)
# data2["specialty"].replace(to_replace='Emergency', value=3, inplace=True, regex=True)
# data2["specialty"].replace(to_replace='General Practice', value=4, inplace=True, regex=True)
# data2["specialty"].replace(to_replace='Internal', value=5, inplace=True, regex=True)
# data2["specialty"].replace(to_replace='Laboratory', value=6, inplace=True, regex=True)
# data2["specialty"].replace(to_replace='Obstetrics and Gynecology', value=7, inplace=True, regex=True)
# data2["specialty"].replace(to_replace='Pathology', value=8, inplace=True, regex=True)
# data2["specialty"].replace(to_replace='Pediatric', value=9, inplace=True, regex=True)
# data2["specialty"].replace(to_replace='Rehabilitation', value=10, inplace=True, regex=True)
# data2["specialty"].replace(to_replace='Surgery', value=10, inplace=True, regex=True)
# data2["specialty"].replace(to_replace='Other', value=11, inplace=True, regex=True)
#
# data2["placesvc"].replace(to_replace='Ambulance', value=1, inplace=True, regex=True)
# data2["placesvc"].replace(to_replace='Home', value=2, inplace=True, regex=True)
# data2["placesvc"].replace(to_replace='Independent Lab', value=3, inplace=True, regex=True)
# data2["placesvc"].replace(to_replace='Inpatient Hospital', value=4, inplace=True, regex=True)
# data2["placesvc"].replace(to_replace='Office', value=5, inplace=True, regex=True)
# data2["placesvc"].replace(to_replace='Outpatient Hospital', value=6, inplace=True, regex=True)
# data2["placesvc"].replace(to_replace='Urgent Care', value=7, inplace=True, regex=True)
# data2["placesvc"].replace(to_replace='Other', value=8, inplace=True, regex=True)
#
# data2['paydelay'].replace(to_replace='162+', value=162, inplace=True, regex=True)
#
# data2['dsfs'].replace(to_replace='0- 1 month', value=1, inplace=True, regex=True)
# data2['dsfs'].replace(to_replace='1- 2 months', value=2, inplace=True, regex=True)
# data2['dsfs'].replace(to_replace='2- 3 months', value=3, inplace=True, regex=True)
# data2['dsfs'].replace(to_replace='3- 4 months', value=4, inplace=True, regex=True)
# data2['dsfs'].replace(to_replace='4- 5 months', value=5, inplace=True, regex=True)
# data2['dsfs'].replace(to_replace='5- 6 months', value=6, inplace=True, regex=True)
# data2['dsfs'].replace(to_replace='6- 7 months', value=7, inplace=True, regex=True)
# data2['dsfs'].replace(to_replace='7- 8 months', value=8, inplace=True, regex=True)
# data2['dsfs'].replace(to_replace='8- 9 months', value=9, inplace=True, regex=True)
# data2['dsfs'].replace(to_replace='9-10 months', value=10, inplace=True, regex=True)
# data2['dsfs'].replace(to_replace='10-11 months', value=11, inplace=True, regex=True)
# data2['dsfs'].replace(to_replace='11-12 months', value=12, inplace=True, regex=True)
#
# data2['PrimaryConditionGroup'].replace(to_replace='AMI', value=1, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='RENAL1', value=34, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='RESPR4', value=35, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='INFEC4', value=36, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='TRAUMA', value=31, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='APPCHOL', value=2, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='ARTHSPIN', value=3, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='HEART4', value=37, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='CANCRA', value=4, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='CANCRB', value=5, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='CATAST', value=7, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='ROAMI', value=38, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='COPD', value=9, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='RENAL2', value=27, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='CHF', value=8, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='METAB1', value=18, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='FLaELEC', value=10, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='FXDISLC', value=39, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='GIBLEED', value=40, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='GIOBSENT', value=41, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='GYNEC1', value=12, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='GYNECA', value=13, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='HIPFX', value=16, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='ODaBNCA', value=23, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='LIVERDZ', value=42, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='MISCL1', value=43, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='MSC2a3', value=21, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='MISCL5', value=44, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='MISCHRT', value=20, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='HEMTOL', value=15, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='HEART2', value=14, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='METAB3', value=45, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='NEUMENT', value=22, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='RENAL3', value=46, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='CANCRM', value=6, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='PNCRDZ', value=47, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='PERVALV', value=24, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='PERINTL', value=24, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='PNEUM', value=48, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='PRGNCY', value=25, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='SEIZURE', value=49, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='SEPSIS', value=29, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='SKNAUT', value=50, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='STROKE', value=30, inplace=True, regex=True)
# data2['PrimaryConditionGroup'].replace(to_replace='UTI', value=32, inplace=True, regex=True)
#
# data2['CharlsonIndex'].replace(to_replace='1-2', value=1, inplace=True, regex=True)
# data2['CharlsonIndex'].replace(to_replace='2-3', value=2, inplace=True, regex=True)
# data2['CharlsonIndex'].replace(to_replace='3-4', value=3, inplace=True, regex=True)
# data2['CharlsonIndex'].replace(to_replace='5+', value=5, inplace=True, regex=True)

#
# data2.to_csv('Claims_Y1.csv', encoding='utf-8', index=False)
# for i in range(len(data3)):
#     if data3['DaysInHospital_Y2'][i] > 0:
#         data3['DaysInHospital_Y2'][i] = 1
#
# data1['sex'].replace(to_replace='M', value=0, inplace=True, regex=True)
# data1['sex'].replace(to_replace='F', value=1, inplace=True, regex=True)
# latest = data2["MemberID"].get(0)
# for i in range(1, len(data2)):
#     print(i)
#     if data2["MemberID"].get(i) == latest:
#         data2.drop(i, inplace=True)
#     else:
#         latest = data2["MemberID"].get(i)
# data1.to_csv('Members_Y1.csv', encoding='utf-8', index=False)
# data2 = data2.join(data1["AgeAtFirstClaim"])
# data2['AgeAtFirstClaim'].replace(to_replace='0-9', value=1, inplace=True, regex=True)
# data2['AgeAtFirstClaim'].replace(to_replace='10-19', value=2, inplace=True, regex=True)
# data2['AgeAtFirstClaim'].replace(to_replace='20-29', value=3, inplace=True, regex=True)
# data2['AgeAtFirstClaim'].replace(to_replace='30-39', value=4, inplace=True, regex=True)
# data2['AgeAtFirstClaim'].replace(to_replace='40-49', value=5, inplace=True, regex=True)
# data2['AgeAtFirstClaim'].replace(to_replace='50-59', value=6, inplace=True, regex=True)
# data2['AgeAtFirstClaim'].replace(to_replace='60-69', value=7, inplace=True, regex=True)
# data2['AgeAtFirstClaim'].replace(to_replace='70-79', value=8, inplace=True, regex=True)
# data2['AgeAtFirstClaim'].replace(to_replace='80+', value=9, inplace=True, regex=True)

# data2.to_csv('Claims_Y1.csv', encoding='utf-8', index=False)
# data3.to_csv('DayInHospital_Y2.csv', encoding='utf-8', index=False)

X_1 = data2.drop(['LengthOfStay', 'MemberID', 'ProviderID', 'vendor', 'pcp', 'Year', 'paydelay'], axis=1)
y_1 = data3['DaysInHospital_Y2']

a_1 = data1['sex']

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

batch_size = 64

MLP = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=11, max_iter=5000, batch_size=batch_size, solver='adam')

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