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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
train_dataset = CustomDataset(X_1_train, y_1_train, a_1_train)
test_dataset = CustomDataset(X_1_test, y_1_test, a_1_test)
valid_dataset = CustomDataset(X_1_val, y_1_val, a_1_val)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

#seed_everything(19)
net = DiabNet(input_size=39, hidden_size=64, num_classes=1)
criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
net.train()
for epoch in range(128):  # 64
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

            # print(label.shape)
            # print(sensitive.shape)

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
        alpha += 1
alpha = alpha/len(female_gt)

for i in male_predic:
    if i == 1:
        beta += 1
beta = beta/len(male_gt)

for i in range(len(female_predic)):
    if female_predic[i] == 1:
        s = np.random.uniform(0, 1)
        if s > (alpha + beta)/(2 * alpha):
            female_predic[i] = 0
            if female_gt[i] == 0:
                correct_10 += 1
                total_10 += 1
                total_11 -= 1
            else:
                correct_11 -= 1
                total_11 -= 1
                total_10 += 1

for i in range(len(male_predic)):
    if male_predic[i] == 0:
        s = np.random.uniform(0, 1)
        if s <= (alpha - beta)/(2 * (1-beta)):
            male_predic[i] = 1
            if male_gt[i] == 1:
                correct_01 += 1
                total_01 += 1
                total_00 -= 1
            else:
                correct_00 -= 1
                total_01 += 1
                total_00 -= 1
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