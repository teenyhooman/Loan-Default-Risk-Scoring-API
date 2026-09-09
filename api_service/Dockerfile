# Loan Default Risk Scoring API - Dockerfile

FROM python:3.12-slim

WORKDIR /app

# نصب dependencies (جدا از کپی کد، برای استفاده‌ی بهتر از Docker layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# کپی کد سرویس
COPY main.py .

# کپی artifact های مدل (preprocessor و مدل نهایی calibrated)
# این فایل‌ها باید قبل از build، در همین پوشه (کنار Dockerfile) قرار داده شوند
# [CHANGED] از مدل uncalibrated (phase9_tuned_...) به مدل calibrated -
# نام فایل زیر را با خروجی واقعی اسکریپت calibration fit تطبیق بده اگر فرق داشت
COPY phase7_preprocessor.joblib .
COPY phase9_calibrated_logistic_regression.joblib .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
