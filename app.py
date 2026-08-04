from fastapi import FastAPI, Request
import joblib
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        print("Received:", data)

        message = data["message"]

        text_vector = vectorizer.transform([message])
        prediction = model.predict(text_vector)[0]

        return {
            "message": message,
            "prediction": str(prediction)
        }

    except Exception as e:
        print("ERROR:", e)
        return {"error": str(e)}