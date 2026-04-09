import random
import numpy as np
import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
df = pd.read_csv(r'recommender/ml/gift_dataset.csv')

clf = joblib.load('recommender/ml/model.pkl')
le_gender_fit = joblib.load('recommender/ml/le_gender.pkl')
df['gender_encoder'] = le_gender_fit.transform(df['gender'])

# prediction function

def generate_wishlist(user_age, user_gender, user_categories, limit=10):
    user_gender_encoded = le_gender_fit.transform([user_gender])[0]

    # attach user details to every gift
    prediction_df = df.copy()
    prediction_df['user_age'] = user_age
    prediction_df['user_gender_encoder'] = user_gender_encoded

    # build feature array
    X_predict = np.array(prediction_df[['category_id', 'gender_encoder', 'min_age', 'max_age', 'user_age', 'user_gender_encoder']])

    # predict
    predictions = clf.predict(X_predict)
    prediction_df['prediction'] = predictions

    # get relevant gifts from selected categories
    results = prediction_df[
        (prediction_df['prediction'] == 1) &
        (prediction_df['category'].isin(user_categories))][['gift_name','category']].to_dict('records')
    

    # fallback to direct filtering if model returns nothing
    if len(results) == 0:
        results = df[
            (df['min_age'] <= user_age) &
            (df['max_age'] >= user_age) &
            ((df['gender'] == user_gender) | (df['gender'] == 'unisex')) &
            (df['category'].isin(user_categories))
        ][['gift_name','category']].to_dict('records')


    # shuffle and limit
    random.shuffle(results)
    return results[:limit]

    


# test it
"""wishlist = generate_wishlist(
    user_age=14,
    user_gender='male',
    user_categories=['Tech', 'Beauty & Fashion'],
    limit=10
)
print(wishlist)"""