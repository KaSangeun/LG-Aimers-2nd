import pandas as pd
import random
import os
import numpy as np

from sklearn.preprocessing import LabelEncoder

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(37) # Seed 고정

# 데이터 불러오기
train_df = pd.read_csv('./data/train.csv')
test_df = pd.read_csv('./data/test.csv')

train_x = train_df.drop(columns=['PRODUCT_ID', 'TIMESTAMP', 'Y_Class', 'Y_Quality'])
train_y = train_df['Y_Class']

test_x = test_df.drop(columns=['PRODUCT_ID', 'TIMESTAMP'])

# 데이터 전처리
train_x = train_x.fillna(0)
test_x = test_x.fillna(0)

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

# Classification Model Fit
RF = RandomForestClassifier(random_state=37)
GBC = GradientBoostingClassifier(random_state=37)
VLD = VotingClassifier(estimators=[('RF',RF), ('GBC',GBC)], voting='soft')
VLD.fit(train_x, train_y)

# Test 데이터에 대한 분류 전, Train 데이터에 대해 학습이 잘 되었는지 확인
print(VLD.score(train_x, train_y))
print('Done.')

# Test 데이터 분류
preds= VLD.predict(test_x)
print('Done.')

# 제출
submit2 = pd.read_csv('./sample_submission.csv')
submit2['Y_Class'] = preds
submit2.to_csv('./baseline_submission4.csv', index=False)

# OS: Windows10
# pandas=1.3.5
# numpy=1.21.6
# sklearn=1.0.2