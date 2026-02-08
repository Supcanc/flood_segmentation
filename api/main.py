from fastapi import FastAPI

app = FastAPI()

@app.get('/predict')
def predict(path: str):
    return path