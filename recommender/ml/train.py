import pandas as pd
import numpy as np
from sklearn import svm, preprocessing,model_selection
import joblib

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
df = pd.read_csv(r'recommender/ml/gift_dataset.csv')

le_gender =  preprocessing.LabelEncoder()
df['gender_encoder'] = le_gender.fit_transform(df['gender'])

sample_users = [
    {'age': 25, 'gender': 'female', 'category': ['Tech', 'Beauty & Fashion']},
    {'age': 15, 'gender': 'male', 'category': ['Entertainment', 'Tech']},
    {'age': 30, 'gender': 'female', 'category': ['Leisure & Sentiment', 'Consumables']},
    {'age': 20, 'gender': 'unisex', 'category': ['Lifestyle & Assets', 'Tech']},
    {'age': 8, 'gender': 'male', 'category': ['Entertainment', 'Consumables']},
    {'age': 50, 'gender': 'female', 'category':['Tech', 'Lifestyle & Assets']},
    {'age': 2, 'gender': 'male', 'category':['Consumables']}
]

rows = [ ]
for user in sample_users:
    temp = df.copy()
    temp['user_age'] = user['age']
    temp['user_gender_encoder'] = le_gender.transform([user['gender']])[0]
    temp['label'] = (
        (temp['min_age'] <= user['age']) &
        (temp['max_age'] >= user['age']) &
        ((temp['gender'] == user['gender']) | (temp['gender'] == 'unisex')) &
        (temp['category'].isin(user['category']))
    ).astype(int)
    rows.append(temp)
training_df = pd.concat(rows, ignore_index=True)

X = np.array(training_df[['category_id', 'gender_encoder', 'min_age', 'max_age', 'user_age', 'user_gender_encoder']])
y = np.array(training_df['label'])

X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y, test_size=0.3)
clf = svm.SVC(kernel='linear')
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
print(accuracy)


joblib.dump(clf,'recommender/ml/model.pkl' )
joblib.dump(le_gender,'recommender/ml/le_gender.pkl' )