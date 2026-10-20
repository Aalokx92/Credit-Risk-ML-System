
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

ml_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_model["model"] = joblib.load(
        BASE_DIR / "credit_risk_model.pkl"
    )

    ml_model["threshold"] = float(
        joblib.load(BASE_DIR / "best_threshold.pkl")
    )

    yield

    ml_model.clear()


app = FastAPI(lifespan=lifespan)


# --------------------------------------------------
# LOAN APPLICATION INPUT SCHEMA
# --------------------------------------------------

class LoanApplication(BaseModel):
    person_age: int
    person_income: float
    person_home_ownership: str
    person_emp_length: float
    loan_intent: str
    loan_grade: str
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_default_on_file: str
    cb_person_cred_hist_length: int


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict(data: LoanApplication):

    # Convert user input into a DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    # Get probability of loan default
    probability = float(
        ml_model["model"].predict_proba(input_df)[0, 1]
    )

    # Apply the saved decision threshold
    threshold = float(ml_model["threshold"])

    prediction = int(probability >= threshold)

    # Return JSON-compatible Python values
    return {
        "default_probability": probability,
        "default_prediction": prediction,
        "threshold": threshold,
        "Result": "High Risk" if prediction == 1 else "Low Risk"
    }


# --------------------------------------------------
# SERVE FRONTEND
# --------------------------------------------------

app.mount(
    "/",
    StaticFiles(directory=str(STATIC_DIR), html=True),
    name="static"
)
