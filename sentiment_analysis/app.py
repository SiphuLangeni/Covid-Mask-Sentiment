from fastapi import FastAPI
from models import MaskSentimentClassifier

app = FastAPI()

@app.post('/classify_mask_sentiment')
async def sentiment_prediction(tweet):
    clf = MaskSentimentClassifier()
    preprocessed_tweet = clf.preprocess(tweet)
    prediction = clf.predict(preprocessed_tweet)
    return {'tweet': tweet, 'label': prediction}

@app.get('/healthcheck', status_code=200)
async def healthcheck():
    return 'Covid mask sentiment analysis app is healthy'
