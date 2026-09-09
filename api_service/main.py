"""
Loan Default Risk Scoring API
Phase 12 - FastAPI Service

مدل: Logistic Regression calibrated (CalibratedClassifierCV, fit شده روی train)
Pipeline پیش‌پردازش: Phase 7 (fit شده روی train)
Threshold عملیاتی پیش‌فرض: 0.10 (بعد از calibration fix، به‌جای 0.35 اصلی Phase 10)

⚠️ نکته‌ی مهم برای مستندسازی: نسخه‌ی calibrated روی همان split_test.csv که
قبلاً یک‌بار (برای نسخه‌ی uncalibrated، threshold=0.35) ارزیابی شده بود،
دوباره ارزیابی شد. این یک انحراف مستند از قانون "test فقط یک‌بار" است -
جزئیات کامل در بخش Methodology Note فایل README آمده. این تصمیم آگاهانه
گرفته شد تا مدل واقعاً calibrated (probability های قابل‌اعتماد) سرو شود.

⚠️ نام فایل مدل (phase9_calibrated_logistic_regression.joblib) بر اساس
مستندات پروژه فرض شده - قبل از deploy واقعی، نام دقیق فایل را با آنچه
اسکریپت calibration fit واقعاً ذخیره کرده تطبیق بده.

این سرویس یک risk score (احتمال default) برمی‌گرداند - نه صرفاً یک تصمیم دودویی -
چون هدف پروژه "Risk Scoring API" است، نه "Risk Decision API".
threshold به‌عنوان پارامتر قابل‌تنظیم در دسترس است.
"""

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

# =========================================================================
# تنظیمات
# =========================================================================
MODEL_DIR = os.environ.get("MODEL_DIR", ".")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "phase7_preprocessor.joblib")
# [CHANGED] از مدل uncalibrated (phase9_tuned_...) به مدل calibrated -
# نام فایل را با خروجی واقعی اسکریپت calibration fit تطبیق بده اگر فرق داشت.
MODEL_PATH = os.path.join(MODEL_DIR, "phase9_calibrated_logistic_regression.joblib")

DEFAULT_THRESHOLD = 0.10  # [CHANGED] از 0.35 به 0.10 - نتیجه‌ی threshold rework بعد از calibration

# مرزهای risk tier - [CHANGED] چون threshold تصمیم از 0.35 به 0.10 منتقل شد، مرزها
# هم باید حول نقطه‌ی تصمیم جدید بازتعریف شوند. این یک پیشنهاد اولیه است، نه یک
# مقدار مستندشده در جایی از پروژه - قبل از استفاده‌ی واقعی، با قضاوت کسب‌وکاری
# (شبیه فرآیندی که برای خود threshold=0.10 طی شد) بازبینی و تأیید شود.
RISK_TIER_BOUNDARIES = {
    "Low": (0.0, 0.10),
    "Medium": (0.10, 0.25),
    "High": (0.25, 1.01),
}

# ستون‌هایی که در Phase 5 حذف شدند اما ممکن است در ورودی خام وجود داشته باشند - نادیده گرفته می‌شوند
COLUMNS_TO_DROP = ["sec_app_earliest_cr_line"]

# مدل‌های global که در startup لود می‌شوند
ml_artifacts: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- Startup ----
    if not os.path.exists(PREPROCESSOR_PATH):
        raise RuntimeError(f"Preprocessor پیدا نشد: {PREPROCESSOR_PATH}")
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model پیدا نشد: {MODEL_PATH}")

    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)

    # استخراج لیست دقیق ستون‌های خام مورد نیاز، مستقیماً از خود preprocessor فیت‌شده
    required_columns = []
    column_types = {}
    for name, _, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        for c in cols:
            required_columns.append(c)
            column_types[c] = "numeric" if name.startswith("num_") else "categorical"

    ml_artifacts["preprocessor"] = preprocessor
    ml_artifacts["model"] = model
    ml_artifacts["required_columns"] = required_columns
    ml_artifacts["column_types"] = column_types

    print(f"مدل و preprocessor با موفقیت لود شدند. تعداد ستون ورودی مورد نیاز: {len(required_columns)}")
    yield
    # ---- Shutdown ----
    ml_artifacts.clear()


app = FastAPI(
    title="Loan Default Risk Scoring API",
    description="پیش‌بینی احتمال default یک وام بر اساس اطلاعات لحظه‌ی application",
    version="1.0.0",
    lifespan=lifespan,
)


# =========================================================================
# Schemas
# =========================================================================
class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(
        ..., description="دیکشنری از نام ستون به مقدار، مطابق با ساختار خام داده‌ی Phase 5"
    )
    threshold: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="threshold تصمیم (اختیاری - در صورت عدم ارسال از مقدار پیش‌فرض 0.35 استفاده می‌شود)"
    )


class PredictionResponse(BaseModel):
    default_probability: float
    risk_tier: str
    threshold_used: float
    decision: str  # "Approve" یا "Review/Reject"


class ModelInfoResponse(BaseModel):
    model_type: str
    default_threshold: float
    risk_tier_boundaries: Dict[str, Any]
    required_feature_count: int
    # [CHANGED] این اعداد قبلاً val_roc_auc/val_pr_auc بودند و مربوط به مدل
    # uncalibrated (Phase 9) بودند (0.7100 / 0.3751). چون معیار validation
    # واقعی مدل calibrated در دسترس نیست (فقط اعداد test داریم، آن‌هم با
    # caveat "دومین نگاه به test" - به README مراجعه کن)، این فیلدها به
    # صراحت به عنوان معیار TEST (نه validation) برچسب‌گذاری شدند تا گمراه‌کننده
    # نباشند.
    test_roc_auc: float
    test_pr_auc: float
    metrics_caveat: str


# =========================================================================
# توابع کمکی
# =========================================================================
def fix_dti_sentinel(df: pd.DataFrame) -> pd.DataFrame:
    """همان قانونی که در Phase 7 روی train/val/test اعمال شد."""
    if "dti" in df.columns:
        df.loc[df["dti"] == 999, "dti"] = np.nan
    return df


def get_risk_tier(probability: float) -> str:
    for tier, (low, high) in RISK_TIER_BOUNDARIES.items():
        if low <= probability < high:
            return tier
    return "High"


def build_input_dataframe(features: Dict[str, Any], required_columns: list) -> pd.DataFrame:
    """یک DataFrame تک‌ردیفی می‌سازد؛ ستون‌های غایب را NaN می‌گذارد (imputer در pipeline هندل می‌کند)."""
    row = {}
    for col in required_columns:
        row[col] = features.get(col, np.nan)
    df = pd.DataFrame([row], columns=required_columns)
    return df


# =========================================================================
# Endpoints
# =========================================================================
@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": "model" in ml_artifacts}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    if "model" not in ml_artifacts:
        raise HTTPException(status_code=503, detail="مدل هنوز لود نشده است.")
    return ModelInfoResponse(
        model_type="Logistic Regression (tuned via RandomizedSearchCV, Phase 9) + CalibratedClassifierCV",
        default_threshold=DEFAULT_THRESHOLD,
        risk_tier_boundaries=RISK_TIER_BOUNDARIES,
        required_feature_count=len(ml_artifacts["required_columns"]),
        test_roc_auc=0.7252,  # از ارزیابی نهایی روی test (به‌صورت دستی درج شده)
        test_pr_auc=0.3977,
        metrics_caveat=(
            "این اعداد از ارزیابی روی test بعد از calibration fix هستند - test "
            "یک‌بار برای مدل uncalibrated و یک‌بار برای این نسخه‌ی calibrated "
            "دیده شده که یک انحراف مستند از پروتکل single-look است (به بخش "
            "Methodology Note در README مراجعه کن)."
        ),
    )


@app.get("/required-fields")
def required_fields():
    if "required_columns" not in ml_artifacts:
        raise HTTPException(status_code=503, detail="مدل هنوز لود نشده است.")
    return {
        "required_columns": ml_artifacts["required_columns"],
        "column_types": ml_artifacts["column_types"],
        "note": "مقادیر NaN/غایب برای ستون‌های numeric با median و برای categorical با category 'Missing' جایگزین می‌شوند - این می‌تواند خودش هم رفتار طبیعی (مثلاً عدم وجود ضامن مشترک) باشد.",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if "model" not in ml_artifacts:
        raise HTTPException(status_code=503, detail="مدل هنوز لود نشده است.")

    preprocessor = ml_artifacts["preprocessor"]
    model = ml_artifacts["model"]
    required_columns = ml_artifacts["required_columns"]

    missing_required = [c for c in required_columns if c not in request.features]
    # توجه: غایب بودن یک ستون خطا نیست (ممکن است NaN طبیعی باشد)، فقط اطلاع‌رسانی می‌شود
    # (برای جلوگیری از spam شدن پاسخ در صورت زیاد بودن، این خط را می‌توان به لاگ منتقل کرد)

    df = build_input_dataframe(request.features, required_columns)
    df = fix_dti_sentinel(df)

    try:
        X_transformed = preprocessor.transform(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"خطا در پردازش ورودی: {str(e)}")

    proba = float(model.predict_proba(X_transformed)[0, 1])
    threshold = request.threshold if request.threshold is not None else DEFAULT_THRESHOLD
    decision = "Review/Reject" if proba >= threshold else "Approve"
    risk_tier = get_risk_tier(proba)

    return PredictionResponse(
        default_probability=round(proba, 4),
        risk_tier=risk_tier,
        threshold_used=threshold,
        decision=decision,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
