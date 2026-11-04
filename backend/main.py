
import re
import html
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# --------------------------------------------------
# 1. PROJECT PATHS AND MODEL LOADING
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "jobshield_model.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)

print("JobShield AI model loaded successfully.")


# --------------------------------------------------
# 2. FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="JobShield AI API",
    description="ML-powered job posting risk analysis API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 3. INPUT DATA FORMAT
# --------------------------------------------------

class JobInput(BaseModel):
    title: str = Field(default="", max_length=1000)
    company_profile: str = Field(default="", max_length=20000)
    description: str = Field(default="", max_length=50000)
    requirements: str = Field(default="", max_length=20000)
    benefits: str = Field(default="", max_length=20000)
    department: str = Field(default="", max_length=2000)
    industry: str = Field(default="", max_length=2000)
    employment_type: str = Field(default="", max_length=2000)
    required_experience: str = Field(default="", max_length=2000)
    required_education: str = Field(default="", max_length=2000)
    function: str = Field(default="", max_length=2000)
    location: str = Field(default="", max_length=2000)


# --------------------------------------------------
# 4. TEXT CLEANING
# --------------------------------------------------

TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "department",
    "industry",
    "employment_type",
    "required_experience",
    "required_education",
    "function",
    "location",
]


def clean_text(value: str) -> str:
    """Remove HTML tags and normalize text."""

    value = html.unescape(value or "")
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def create_combined_text(job: JobInput) -> str:
    """
    Create text in the same labelled-field format
    used during ML model training.
    """

    sections = []

    for column in TEXT_COLUMNS:
        value = clean_text(getattr(job, column, ""))

        if value:
            label = column.replace("_", " ").title()
            sections.append(f"{label}: {value}")

    return " ".join(sections)


# --------------------------------------------------
# 5. HOME API
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Welcome to JobShield AI API",
        "status": "running",
        "model_loaded": True,
        "version": "1.0.0"
    }


# --------------------------------------------------
# 6. HEALTH CHECK API
# --------------------------------------------------

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH)
    }


# --------------------------------------------------
# 7. JOB ANALYSIS API
# --------------------------------------------------

@app.post("/analyze")
def analyze_job(job: JobInput):

    try:
        combined_text = create_combined_text(job)

        if not combined_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Please provide at least a job title or job description."
            )

        # Generate ML prediction
        prediction = int(model.predict([combined_text])[0])

        # Get the probability of the fraudulent class
        probabilities = model.predict_proba([combined_text])[0]

        classes = list(model.classes_)

        if 1 in classes:
            fraud_index = classes.index(1)
            risk_probability = float(probabilities[fraud_index])
        else:
            raise RuntimeError(
                "The trained model does not contain the fraud class (1)."
            )

        risk_score = round(risk_probability * 100)

        if prediction == 1:
            result = "Potentially Suspicious"
            explanation = (
                "The ML model identified patterns associated "
                "with fraudulent job postings."
            )
        else:
            result = "Likely Legitimate"
            explanation = (
                "The ML model identified patterns more commonly "
                "associated with legitimate job postings."
            )

        return {
            "success": True,
            "project": "JobShield AI",
            "prediction": prediction,
            "result": result,
            "risk_score": risk_score,
            "risk_probability": round(risk_probability, 4),
            "explanation": explanation,
            "disclaimer": (
                "This is an ML-based risk estimate, not proof that "
                "a job or employer is genuine or fraudulent. "
                "Verify the employer independently."
            )
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Job analysis failed: {str(error)}"
        )


# --------------------------------------------------
# 8. RUNNING INFORMATION
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )
