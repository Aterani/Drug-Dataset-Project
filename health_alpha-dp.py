import random
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from CustomDataset import CustomDataset
from DiabNet import DiabNet
from sklearn.model_selection import train_test_split
import pandas as pd

from FullyConnectedNet import FullyConnectedNet


def seed_everything(seed):
    """
    Changes the seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_1 = data2.drop(['LengthOfStay', 'MemberID', 'ProviderID', 'vendor', 'pcp', 'Year', 'paydelay'], axis=1)
y_1 = data3['DaysInHospital_Y2']

a_1 = data1['sex']

X_1_train, X_1_temp, y_1_train, y_1_temp, a_1_train, a_1_temp = train_test_split(X_1, y_1, a_1, test_size=0.4,
                                                                                 random_state=42)
X_1_test, X_1_val, y_1_test, y_1_val, a_1_test, a_1_val = train_test_split(X_1_temp, y_1_temp, a_1_temp, test_size=0.5,
                                                                           random_state=42)

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
train_dataset = CustomDataset(X_1_train, y_1_train, a_1_train)
test_dataset = CustomDataset(X_1_test, y_1_test, a_1_test)
valid_dataset = CustomDataset(X_1_val, y_1_val, a_1_val)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

#seed_everything(0)
net = FullyConnectedNet(input_size=6, hidden_size=64, num_classes=1)
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
net.train()
for epoch in range(64):  # 64
    for batch in train_dataloader:
        data1, labels, sensitive = batch[0], batch[1], batch[2]
        optimizer.zero_grad()
        outputs = net(data1)
        labels = labels.unsqueeze(1)
        sensitive = sensitive.unsqueeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

criteria = nn.BCEWithLogitsLoss()
net.eval()
test_pred = []
test_gt = []
sense_gt = []
female_predic = []
female_gt = []
male_predic = []
male_gt = []
correct_00, total_00 = 0, 0
correct_01, total_01 = 0, 0
correct_10, total_10 = 0, 0
correct_11, total_11 = 0, 0
epoch_loss = 0
with torch.no_grad():
    with tqdm(test_dataloader, unit="batch") as tepoch:
        for content in tepoch:
            test, label, sensitive = content
            test = test.to(device)
            label = label.unsqueeze(1)
            prediction = net(test)
            label = label.to(torch.float).to(device)
            loss = criteria(prediction, label)
            epoch_loss += loss.item()
            prediction = torch.sigmoid(prediction)
            gt = label.detach().cpu().numpy()
            sen = sensitive.detach().cpu().numpy()
            test_pred.extend(torch.round(prediction.squeeze(1)).detach().cpu().numpy())
            test_gt.extend(gt)
            sense_gt.extend(sen)
            label = label.squeeze()

            mask_00 = ((label == 0) & (sensitive == 0))
            mask_01 = ((label == 0) & (sensitive == 1))
            mask_10 = ((label == 1) & (sensitive == 0))
            mask_11 = ((label == 1) & (sensitive == 1))

            predic = torch.round(prediction.squeeze(1)).detach().cpu()
            correct_00 += (predic[mask_00] == label[mask_00]).float().sum().item()
            total_00 += mask_00.float().sum().item()

            correct_01 += (predic[mask_01] == label[mask_01]).float().sum().item()
            total_01 += mask_01.float().sum().item()

            correct_10 += (predic[mask_10] == label[mask_10]).float().sum().item()
            total_10 += mask_10.float().sum().item()

            correct_11 += (predic[mask_11] == label[mask_11]).float().sum().item()
            total_11 += mask_11.float().sum().item()
        epoch_loss = epoch_loss / len(test_dataloader)
    acc_00 = correct_00 / total_00
    acc_01 = correct_01 / total_01
    acc_10 = correct_10 / total_10
    acc_11 = correct_11 / total_11

    print(f'Accuracy for y=0, s=0: {acc_00}', total_00)
    print(f'Accuracy for y=0, s=1: {acc_01}', total_01)
    print(f'Accuracy for y=1, s=0: {acc_10}', total_10)
    print(f'Accuracy for y=1, s=1: {acc_11}', total_11)
    for i in range(len(sense_gt)):
        if sense_gt[i] == 1:
            female_predic.append(test_pred[i])
            female_gt.append(test_gt[i])
        else:
            male_predic.append(test_pred[i])
            male_gt.append(test_gt[i])

    female_CM = confusion_matrix(female_gt, female_predic)
    male_CM = confusion_matrix(male_gt, male_predic)
    print(female_CM)
    print(male_CM)
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
    print('acc', accuracy_score(test_gt, test_pred))
    print(accuracy_score(test_gt, test_pred), epoch_loss)

alpha = 0
beta = 0
for i in female_predic:
    if i == 1:
        beta += 1
beta = beta/len(female_gt)

for i in male_predic:
    if i == 1:
        alpha += 1
alpha = alpha/len(male_gt)

for i in range(len(male_predic)):
    if male_predic[i] == 1:
        s = np.random.uniform(0, 1)
        if s > (alpha + beta)/(2 * alpha):
            male_predic[i] = 0

for i in range(len(female_predic)):
    if female_predic[i] == 0:
        s = np.random.uniform(0, 1)
        if s <= (alpha - beta)/(2 * (1-beta)):
            female_predic[i] = 1

female = 0
male = 0
for i in range(len(sense_gt)):
    if sense_gt[i] == 1:
        test_pred[i] = female_predic[female]
        female += 1
    else:
        test_pred[i] = male_predic[male]
        male += 1
print("after fairness:", alpha, beta)
acc_00 = correct_00 / total_00
acc_01 = correct_01 / total_01
acc_10 = correct_10 / total_10
acc_11 = correct_11 / total_11

print(f'Accuracy for y=0, s=0: {acc_00}', total_00)
print(f'Accuracy for y=0, s=1: {acc_01}', total_01)
print(f'Accuracy for y=1, s=0: {acc_10}', total_10)
print(f'Accuracy for y=1, s=1: {acc_11}', total_11)

female_CM = confusion_matrix(female_gt, female_predic)
male_CM = confusion_matrix(male_gt, male_predic)
print(female_CM)
print(male_CM)
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
print('acc', accuracy_score(test_gt, test_pred))
print(accuracy_score(test_gt, test_pred), epoch_loss)
print(confusion_matrix(test_gt, test_pred))
