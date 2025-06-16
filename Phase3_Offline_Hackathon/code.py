# Import
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import xgboost as xgb
import test
import math

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(37) # Seed 고정

# 데이터 불러오기
train_df = pd.read_csv('./train.csv')
test_df = pd.read_csv('./test.csv')

train_x = train_df.drop(columns=['PRODUCT_ID', 'Y_Class', 'Y_Quality'])
train_y = train_df['Y_Class']

test_x = test_df.drop(columns=['PRODUCT_ID'])

# 데이터 전처리
train_x = train_x.fillna(-1) #빈 값을 -1값으로 처리
test_x = test_x.fillna(-1) #빈 값을 -1값으로 처리


# qualitative to quantitative
qual_col = ['LINE', 'PRODUCT_CODE']

for i in qual_col:
    le = LabelEncoder()
    le = le.fit(train_x[i])
    train_x[i] = le.transform(train_x[i])

    for label in np.unique(test_x[i]):
        if label not in le.classes_:
            le.classes_ = np.append(le.classes_, label)
    test_x[i] = le.transform(test_x[i])
print('Done.')

# Classification Model Import
from sklearn.ensemble import RandomForestClassifier #예시코드
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

# Classification Model Fit
RF = RandomForestClassifier(random_state=37)
GBC = GradientBoostingClassifier(random_state=37)
KNN = KNeighborsClassifier(n_neighbors = 3)
ADB = AdaBoostClassifier(n_estimators=150, random_state=37)
VLD = VotingClassifier(estimators=[('RF',RF), ('GBC',GBC), ('KNN', KNN), ('ADB', ADB)], voting='soft', weights=[1,3,2,1], n_jobs=-1)
VLD.fit(train_x*train_x, train_y)

# Test 데이터에 대한 분류 전, Train 데이터에 대해 학습이 잘 되었는지 확인
print(VLD.score(train_x*train_x, train_y))
print('Done.')

# Test 데이터 분류
preds= VLD.predict(test_x*test_x)
print('Done.')

# 제출
submit2 = pd.read_csv('./sample_submission.csv')
submit2['Y_Class'] = preds
submit2.to_csv('./jm21.csv', index=False)