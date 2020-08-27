from fastapi import FastAPI


app = FastAPI()


@app.get('/healthcheck', status_code=200)
async def healthcheck():
    return 'Covid mask sentiment app is healthy!'
