from fastapi import FastAPI
import mask_classifier_router

app = FastAPI()
app.include_router(mask_classifier_router, prefix='/sentiment_analysis')

@app.get('/healthcheck', status_code=200)
async def healthcheck():
    return 'Mask classifier is ready to go.'