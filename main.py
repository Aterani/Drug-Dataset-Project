import random

import torch
from torch._C import device
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from CustomDataset import CustomDataset
from FullyConnectedNet import FullyConnectedNet
from sklearn.model_selection import train_test_split
from fairlearn.postprocessing import ThresholdOptimizer, plot_threshold_optimizer
from sklearn.utils.estimator_checks import check_estimator

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def seed_everything(seed):
    """
    Changes the seed for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
train_dataset = CustomDataset(X_1_train, y_1_train, a_1_train)
test_dataset = CustomDataset(X_1_test, y_1_test, a_1_test)
valid_dataset = CustomDataset(X_1_val, y_1_val, a_1_val)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

seed_everything(0)
net = FullyConnectedNet(input_size=12, hidden_size=128, num_classes=1)
criterion = nn.BCEWithLogitsLoss()

# optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
weight_decay=1e-3
init_lr=1e-3
momentum_decay = 0.9
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
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
    epoch_loss = epoch_loss / len(test_dataloader)

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
