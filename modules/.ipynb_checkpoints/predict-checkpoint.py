import pandas as pd
import joblib

def predict_sentiment(data, review_col='Full Review'):
    model = joblib.load("./models/model.pkl")
    preds = model.predict(data[review_col])
    labels = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    data['Sentiment'] = [labels[p] if p in labels else 'Unknown' for p in preds]
    return data
