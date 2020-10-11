from fastapi import APIRouter
from sentiment_classifier.mask_classifier import MaskClassifier
from starlette.responses import JSONResponse

router = APIRouter()

@router.post('/classify_mask_sentiment')
def get_sentiment(tweet: str):
    clf = MaskClassifier()
    return JSONResponse(clf.get_predictions(tweet))