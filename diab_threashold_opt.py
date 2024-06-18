import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import sklearn.neural_network


data = pd.read_csv("diabetic_data.csv")
# for i in range(len(data)):
#     if data["race"][i] == '?':
#         data.drop(i, inplace=True)

# data["age"].replace(to_replace='\\[0-10\\)', value=10, inplace=True, regex=True)
# data["age"].replace(to_replace='\\[10-20\\)', value=20, inplace=True, regex=True)
# data["age"].replace(to_replace='\\[20-30\\)', value=30, inplace=True, regex=True)
# data["age"].replace(to_replace='\\[30-40\\)', value=40, inplace=True, regex=True)
# data["age"].replace(to_replace='\\[40-50\\)', value=50, inplace=True, regex=True)
# data["age"].replace(to_replace='\\[50-60\\)', value=60, inplace=True, regex=True)
# data["age"].replace(to_replace='\\[60-70\\)', value=70, inplace=True, regex=True)
# data["age"].replace(to_replace='\\[70-80\\)', value=80, inplace=True, regex=True)
# data["age"].replace(to_replace='\\[80-90\\)', value=90, inplace=True, regex=True)
# data["age"].replace(to_replace='\\[90-100\\)', value=100, inplace=True, regex=True)
#
# data["max_glu_serum"].replace(to_replace='>200', value=1, inplace=True, regex=True)
# data["max_glu_serum"].replace(to_replace='>300', value=2, inplace=True, regex=True)
# data["max_glu_serum"].replace(to_replace='Norm', value=3, inplace=True, regex=True)
#data["max_glu_serum"].replace(to_replace='', value=4, inplace=True, regex=True)
#
# data["A1Cresult"].replace(to_replace='>8', value=1, inplace=True, regex=True)
# data["A1Cresult"].replace(to_replace='>7', value=2, inplace=True, regex=True)
# data["A1Cresult"].replace(to_replace='Norm', value=3, inplace=True, regex=True)
#data["A1Cresult"].replace(to_replace='', value=4, inplace=True, regex=True)
#
# data["metformin"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["metformin"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["metformin"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["metformin"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["repaglinide"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["repaglinide"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["repaglinide"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["repaglinide"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["nateglinide"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["nateglinide"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["nateglinide"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["nateglinide"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["chlorpropamide"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["chlorpropamide"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["chlorpropamide"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["chlorpropamide"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["glimepiride"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["glimepiride"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["glimepiride"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["glimepiride"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["acetohexamide"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["acetohexamide"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["acetohexamide"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["acetohexamide"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["glipizide"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["glipizide"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["glipizide"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["glipizide"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["glyburide"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["glyburide"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["glyburide"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["glyburide"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["tolbutamide"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["tolbutamide"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["tolbutamide"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["tolbutamide"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["pioglitazone"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["pioglitazone"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["pioglitazone"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["pioglitazone"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["rosiglitazone"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["rosiglitazone"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["rosiglitazone"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["rosiglitazone"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["acarbose"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["acarbose"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["acarbose"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["acarbose"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["miglitol"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["miglitol"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["miglitol"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["miglitol"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["troglitazone"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["troglitazone"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["troglitazone"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["troglitazone"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["tolazamide"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["tolazamide"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["tolazamide"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["tolazamide"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["examide"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["examide"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["examide"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["examide"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["citoglipton"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["citoglipton"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["citoglipton"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["citoglipton"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["insulin"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["insulin"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["insulin"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["insulin"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["glyburide-metformin"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["glyburide-metformin"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["glyburide-metformin"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["glyburide-metformin"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["glipizide-metformin"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["glipizide-metformin"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["glipizide-metformin"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["glipizide-metformin"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["glimepiride-pioglitazone"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["glimepiride-pioglitazone"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["glimepiride-pioglitazone"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["glimepiride-pioglitazone"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["metformin-rosiglitazone"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["metformin-rosiglitazone"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["metformin-rosiglitazone"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["metformin-rosiglitazone"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["metformin-pioglitazone"].replace(to_replace='Up', value=1, inplace=True, regex=True)
# data["metformin-pioglitazone"].replace(to_replace='Down', value=2, inplace=True, regex=True)
# data["metformin-pioglitazone"].replace(to_replace='Steady', value=3, inplace=True, regex=True)
# data["metformin-pioglitazone"].replace(to_replace='No', value=4, inplace=True, regex=True)
#
# data["change"].replace(to_replace='No', value=0.0, inplace=True, regex=True)
# data["change"].replace(to_replace='Ch', value=1.0, inplace=True, regex=True)
#
# data["diabetesMed"].replace(to_replace='No', value=0.0, inplace=True, regex=True)
# data["diabetesMed"].replace(to_replace='Yes', value=1.0, inplace=True, regex=True)
#
# data["readmitted"].replace(to_replace='NO', value='0.0', inplace=True, regex=True)
# data["readmitted"].replace(to_replace='<30', value='1.0', inplace=True, regex=True)
# data["readmitted"].replace(to_replace='>30', value='1.0', inplace=True, regex=True)
# data["race"].replace(to_replace='Caucasian', value=1, inplace=True, regex=True)
# data["race"].replace(to_replace='AfricanAmerican', value=2, inplace=True, regex=True)
# data["race"].replace(to_replace='Asian', value=3, inplace=True, regex=True)
# data["race"].replace(to_replace='Other', value=5, inplace=True, regex=True)
# data["race"].replace(to_replace='\\?', value=5, inplace=True, regex=True)
# data["race"].replace(to_replace='Hispanic', value=4, inplace=True, regex=True)
data.to_csv('diabetic_data.csv', encoding='utf-8', index=False)

X_1 = data.drop(['encounter_id', 'patient_nbr', 'readmitted', 'weight', 'payer_code', 'medical_specialty'
                 , 'diag_1', 'diag_2', 'diag_3', 'max_glu_serum', 'A1Cresult'], axis=1)
y_1 = data['readmitted']

a_1 = data['gender']


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