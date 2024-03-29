from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

data = pd.read_csv("/Users/aneeshterani/Downloads/drug_consumption.data")
# print(data.columns)]
data["Coke"] = pd.Series(data["Coke"], dtype=pd.StringDtype())

data["Gender"].replace(to_replace=0.48246, value=1, inplace=True, regex=True)
data["Gender"].replace(to_replace=-0.48246, value=0, inplace=True, regex=True)
data["Coke"].replace(to_replace='CL0', value='0', inplace=True, regex=True)
data["Coke"].replace(to_replace='CL1', value='0', inplace=True, regex=True)
data["Coke"].replace(to_replace='CL2', value='1', inplace=True, regex=True)
data["Coke"].replace(to_replace='CL3', value='1', inplace=True, regex=True)
data["Coke"].replace(to_replace='CL4', value='1', inplace=True, regex=True)
data["Coke"].replace(to_replace='CL5', value='1', inplace=True, regex=True)
data["Coke"].replace(to_replace='CL6', value='1', inplace=True, regex=True)

# print(data["Coke"])
data.to_csv('drug_data.csv', encoding='utf-8', index=False)
data = pd.read_csv('drug_data.csv')

data_1 = pd.read_csv('drug_data.csv')
for i in range(len(data_1)):
    if data_1['Gender'][i] == 1:
        data_1.drop(i, inplace=True)

data_2 = pd.read_csv('drug_data.csv')
for i in range(len(data_2)):
    if data_2['Gender'][i] == 0:
        data_2.drop(i, inplace=True)

# plt.figure(figsize=(8, 6))
# sns.histplot(x=data["Coke"], y=data["Gender"])
# maledrug = 0
# femaledrug = 0
# malenon = 0
# femalenon = 0
# for i in range(len(data["Coke"])):
#     if data["Coke"][i] == 0 and data["Gender"][i] == -0.48246:
#         malenon += 1
#     elif data["Coke"][i] == 0 and data["Gender"][i] == 0.48246:
#         femalenon += 1
#     elif data["Coke"][i] == 1 and data["Gender"][i] == -0.48246:
#         maledrug += 1
#     else:
#         femaledrug += 1
# # plt.show()
# print(str(maledrug) + "\n" + str(femaledrug) + "\n" + str(malenon) + "\n" + str(femalenon) + "\n")


def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        #
        # print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        # print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred, labels=[1,0])}\n")

    elif not train:
        pred = clf.predict(X_test)
        # clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        # print("Test Result:\n================================================")
        # print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        # print("_______________________________________________")
        # print(f"CLASSIFICATION REPORT:\n{clf_report}")
        # print("_______________________________________________")
        # print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred, labels=[1,0])}\n")


X_1 = data_1.drop(
    ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin',
     'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA'], axis=1)
y_1 = data_1['Coke']

a_1 = data_1['Gender']

X_2 = data_2.drop(
    ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin',
     'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA'], axis=1)
y_2 = data_2['Coke']

a_2 = data_2['Gender']

x_3 = data.drop(['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke', 'Crack', 'Ecstasy', 'Heroin',
     'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA'], axis=1)
y_3 = data['Coke']

X_1_train, X_1_temp, y_1_train, y_1_temp = train_test_split(X_1, y_1, test_size=0.4, random_state=42)
X_1_test, X_1_val, y_1_test, y_1_val = train_test_split(X_1_temp, y_1_temp, test_size=0.5, random_state=42)

x_3_train, x_3_temp, y_3_train, y_3_temp = train_test_split(x_3, y_3, test_size=0.4, random_state=42)
x_3_test, x_3_val, y_3_test, y_3_val = train_test_split(x_3_temp, y_3_temp, test_size=0.5, random_state=42)

X_2_train, X_2_temp, y_2_train, y_2_temp = train_test_split(X_2, y_2, test_size=0.4, random_state=42)
X_2_test, X_2_val, y_2_test, y_2_val = train_test_split(X_2_temp, y_2_temp, test_size=0.5, random_state=42)

# cat_columns = []
num_columns = ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore',
               'Impulsive', 'SS']

ct = make_column_transformer(
    (MinMaxScaler(), num_columns),
    (StandardScaler(), num_columns),
    remainder='passthrough'
)

X_1_train = ct.fit_transform(X_1_train)
X_1_test = ct.transform(X_1_test)
X_1_val = ct.transform(X_1_val)

X_2_train = ct.fit_transform(X_2_train)
X_2_test = ct.transform(X_2_test)
X_2_val = ct.transform(X_2_val)

x_3_train = ct.fit_transform(x_3_train)
x_3_test = ct.transform(x_3_test)
x_3_val = ct.transform(x_3_val)

lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_1_train, y_1_train)

print_score(lr_clf, X_1_train, y_1_train, X_1_test, y_1_test, train=True)
print_score(lr_clf, X_1_train, y_1_train, X_1_test, y_1_test, train=False)

lr_clf.fit(X_2_train, y_2_train)

print_score(lr_clf, X_2_train, y_2_train, X_2_test, y_2_test, train=True)
print_score(lr_clf, X_2_train, y_2_train, X_2_test, y_2_test, train=False)

lr_clf.fit(x_3_train, y_3_train)
print_score(lr_clf, x_3_train, y_3_train, x_3_test, y_3_test, train=True)
print_score(lr_clf, x_3_train, y_3_train, x_3_test, y_3_test, train=False)