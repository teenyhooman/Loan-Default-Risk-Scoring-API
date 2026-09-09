"""
================================================================================
Loan Default Risk Scoring API — Full Pipeline (All Phases, Single File)
================================================================================
این فایل تمام کد فازهای پروژه را به ترتیب اجرا، پشت سر هم قرار می‌دهد.
هدف این فایل صرفاً "مرور/تحویل ساده" است، نه یک اجرای بهینه‌شده؛ به همین دلیل
هر فاز دقیقاً همان کاری را می‌کند که به‌صورت مستقل انجام می‌داد (شامل بارگذاری
مجدد فایل‌های میانی از دیسک بین فازها) — این با نحوه‌ی واقعی اجرای پروژه سازگار
است و برای review/دفاع شفاف‌تر است.

⚠️ نکات مهم قبل از اجرا:
  1. تمام مسیرها به یک پوشه‌ی واحد (BASE_DIR پایین) یکدست شدند. در نسخه‌ی
     اصلی، Phase 10/11 مسیر را هاردکد کرده بودند (D:/aashit/proccess) و
     calibration_check.py حتی به دو پوشه‌ی متفاوت اشاره می‌کرد
     (D:/aashit/New folder در مقابل D:/aashit/proccess) — این ناسازگاری در
     نسخه‌ی اصلی باید قبل از اعتماد به نتایج calibration بررسی شود.
  2. بخش "Calibration Fix" (فیت کردن CalibratedClassifierCV و ساخت
     phase9_calibrated_logistic_regression.joblib) در فایل‌های دریافتی موجود
     نبود — فقط اسکریپت چک اولیه (calibration_check.py) موجود بود. آن بخش
     را باید بعداً اضافه کرد؛ یک placeholder مشخص پایین گذاشته شده.
  3. در Phase 9، جستجوی Logistic Regression شامل هایپرپارامتر l1_ratio است
     ولی چون penalty صریحاً 'elasticnet' تنظیم نشده (پیش‌فرض 'l2' است)،
     l1_ratio عملاً در جستجو بی‌اثر بوده. این باگ در همین فایل *برطرف نشده*
     تا فایل دقیقاً منعکس‌کننده‌ی کدی باشد که واقعاً اجرا و نتایجش گزارش شده؛
     در صورت اجرای مجدد پروژه، باید penalty="elasticnet" هم اضافه شود.
  4. اجرای کامل این فایل از ابتدا تا انتها روی کل pipeline (به‌خصوص
     RandomizedSearchCV در Phase 9) می‌تواند طولانی باشد. برای اجرای تکه‌ای
     از یک فاز مشخص، به بخش‌های جداشده با کامنت‌های "PHASE N" مراجعه کنید.

هر فاز دقیقاً با همان محتوایی که به‌صورت مستقل نوشته و اجرا شده، اینجا آمده؛
فقط مسیرها یکدست شدند (نشانه‌گذاری‌شده با کامنت‌های [PATH FIX]).
================================================================================
"""

import os

# ==============================================================================
# CONFIG مشترک برای همه‌ی فازها -- این تنها بخشی است که باید قبل از اجرا ویرایش شود
# ==============================================================================
BASE_DIR = "."   # پوشه‌ای که همه‌ی فایل‌های CSV/joblib/npz پروژه در آن هستند
FINAL_THRESHOLD = 0.35   # threshold عملیاتی منتخب Phase 10 (قبل از calibration fix)


# ##############################################################################
# PHASE 2 — Target Construction
# ##############################################################################
"""
Phase 2 — Target construction.
تصمیم: class 0 = Fully Paid, class 1 = Charged Off + Default (merge شده).
Current/Late/In Grace Period و گروه "does not meet credit policy" حذف می‌شوند.
"""
import pandas as pd

def run_phase2():
    CSV_PATH = os.path.join(BASE_DIR, "lending_club_loan_data.csv")  # فایل خام ~1GB
    OUT_PATH = os.path.join(BASE_DIR, "loan_data_labeled.csv")
    CHUNK_SIZE = 50_000

    KEEP_STATUSES_POSITIVE = {"Charged Off", "Default"}
    KEEP_STATUSES_NEGATIVE = {"Fully Paid"}
    KEEP_STATUSES = KEEP_STATUSES_POSITIVE | KEEP_STATUSES_NEGATIVE

    reader = pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE, low_memory=False)

    first_chunk = True
    total_written = 0
    status_counts_before, status_counts_after = {}, {}

    for chunk in reader:
        for status, c in chunk["loan_status"].value_counts().items():
            status_counts_before[status] = status_counts_before.get(status, 0) + c

        mask = chunk["loan_status"].isin(KEEP_STATUSES)
        filtered = chunk.loc[mask].copy()
        filtered["target"] = filtered["loan_status"].map(
            lambda s: 1 if s in KEEP_STATUSES_POSITIVE else 0
        )
        for status, c in filtered["loan_status"].value_counts().items():
            status_counts_after[status] = status_counts_after.get(status, 0) + c

        filtered.to_csv(OUT_PATH, mode="w" if first_chunk else "a",
                         header=first_chunk, index=False)
        first_chunk = False
        total_written += len(filtered)

    print("Pre-filter:", pd.Series(status_counts_before).sort_values(ascending=False))
    print("Post-filter:", pd.Series(status_counts_after).sort_values(ascending=False))
    print(f"Rows written: {total_written:,} -> {OUT_PATH}")


# ##############################################################################
# PHASE 3 — Stratified Sampling + EDA
# ##############################################################################
"""
Phase 3a — نمونه‌گیری stratified به 100,000 ردیف (seed=42، حفظ class balance).
Phase 3b — EDA کامل روی sample (ساختار، missing values، anomalyها، رابطه‌ی
           feature-target). جزئیات کامل EDA برای اختصار اینجا تکرار نشده -
           فقط مرحله‌ی نمونه‌گیری که ورودی فازهای بعدی است آمده.
"""
import numpy as np

def run_phase3_sampling():
    IN_PATH = os.path.join(BASE_DIR, "loan_data_labeled.csv")
    OUT_PATH = os.path.join(BASE_DIR, "loan_data_sample_100k.csv")
    CHUNK_SIZE = 50_000
    TARGET_SAMPLE_SIZE = 100_000
    SEED = 42
    TOTAL_ROWS = 1_303_638
    SAMPLE_FRAC = TARGET_SAMPLE_SIZE / TOTAL_ROWS

    rng = np.random.RandomState(SEED)
    reader = pd.read_csv(IN_PATH, chunksize=CHUNK_SIZE, low_memory=False)

    first_chunk = True
    total_written = 0
    class_counts = {0: 0, 1: 0}

    for chunk in reader:
        for cls in (0, 1):
            sub = chunk[chunk["target"] == cls]
            if len(sub) == 0:
                continue
            sampled = sub.sample(frac=SAMPLE_FRAC, random_state=rng)
            class_counts[cls] += len(sampled)
            sampled.to_csv(OUT_PATH, mode="w" if first_chunk else "a",
                            header=first_chunk, index=False)
            first_chunk = False
            total_written += len(sampled)

    print(f"Total rows written: {total_written:,}")
    print(f"Class 0: {class_counts[0]:,} | Class 1: {class_counts[1]:,}")
    print(f"-> {OUT_PATH}")


# ##############################################################################
# PHASE 4 + 5 — Leakage Audit + Final Feature Selection
# ##############################################################################
"""
Phase 4 — Leakage audit (تصمیم‌گیری مستند در README/چت، نه صرفاً کد).
Phase 5 — اعمال تصمیمات: حذف 57 ستون (Group D/C/B + scope/quality + redundant)،
          ساخت credit_history_length_months، تولید loan_data_clean_features.csv
          (100,004 x 90 = 89 feature + target).
"""

def run_phase5():
    IN_PATH = os.path.join(BASE_DIR, "loan_data_sample_100k.csv")
    OUT_PATH = os.path.join(BASE_DIR, "loan_data_clean_features.csv")

    df = pd.read_csv(IN_PATH, low_memory=False)

    GROUP_D = [
        "total_pymnt", "total_pymnt_inv", "total_rec_prncp", "total_rec_int",
        "total_rec_late_fee", "recoveries", "collection_recovery_fee",
        "out_prncp", "out_prncp_inv", "last_pymnt_d", "last_pymnt_amnt",
        "next_pymnt_d", "last_credit_pull_d",
    ]
    GROUP_C = [
        "hardship_flag", "hardship_type", "hardship_reason", "hardship_status",
        "deferral_term", "hardship_amount", "hardship_start_date", "hardship_end_date",
        "payment_plan_start_date", "hardship_length", "hardship_dpd",
        "hardship_loan_status", "orig_projected_additional_accrued_interest",
        "hardship_payoff_balance_amount", "hardship_last_payment_amount",
        "debt_settlement_flag", "debt_settlement_flag_date", "settlement_status",
        "settlement_date", "settlement_amount", "settlement_percentage",
        "settlement_term", "pymnt_plan", "acc_now_delinq", "delinq_amnt",
        "disbursement_method",
    ]
    GROUP_B = [
        "funded_amnt", "funded_amnt_inv", "int_rate", "installment",
        "grade", "sub_grade", "initial_list_status", "policy_code",
    ]
    SCOPE_DROPS = ["id", "member_id", "url", "desc", "emp_title", "title", "zip_code"]
    REDUNDANT = ["loan_status"]

    ALL_DROPS = GROUP_D + GROUP_C + GROUP_B + SCOPE_DROPS + REDUNDANT

    issue_dt = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    cr_dt = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    df["credit_history_length_months"] = (
        (issue_dt.dt.year - cr_dt.dt.year) * 12 + (issue_dt.dt.month - cr_dt.dt.month)
    )
    ALL_DROPS += ["earliest_cr_line", "issue_d"]

    clean_df = df.drop(columns=[c for c in ALL_DROPS if c in df.columns])
    print("Final shape:", clean_df.shape)
    clean_df.to_csv(OUT_PATH, index=False)
    print(f"-> {OUT_PATH}")


# ##############################################################################
# PHASE 6 — Train / Validation / Test Split
# ##############################################################################
from sklearn.model_selection import train_test_split

def run_phase6():
    IN_PATH = os.path.join(BASE_DIR, "loan_data_clean_features.csv")
    SEED = 42

    df = pd.read_csv(IN_PATH, low_memory=False)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED
    )
    val_fraction_of_temp = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_fraction_of_temp, stratify=y_temp, random_state=SEED
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    val_df = pd.concat([X_val, y_val], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(os.path.join(BASE_DIR, "split_train.csv"), index=False)
    val_df.to_csv(os.path.join(BASE_DIR, "split_val.csv"), index=False)
    test_df.to_csv(os.path.join(BASE_DIR, "split_test.csv"), index=False)
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")


# ##############################################################################
# PHASE 7 — Preprocessing Pipeline  [PATH FIX: DATA_DIR/OUTPUT_DIR -> BASE_DIR]
# ##############################################################################
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PowerTransformer

def run_phase7():
    TRAIN_PATH = os.path.join(BASE_DIR, "split_train.csv")
    VAL_PATH = os.path.join(BASE_DIR, "split_val.csv")
    TEST_PATH = os.path.join(BASE_DIR, "split_test.csv")
    TARGET_COL = "target"

    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    print(f"Train: {train_df.shape} | Val: {val_df.shape} | Test: {test_df.shape}")
    assert train_df.shape[1] == 90, "ورودی train باید 90 ستون داشته باشد!"

    def fix_dti_sentinel(df):
        df = df.copy()
        if "dti" in df.columns:
            n_sentinel = (df["dti"] == 999).sum()
            if n_sentinel > 0:
                print(f"  -> {n_sentinel} مقدار dti=999 به NaN تبدیل شد.")
            df.loc[df["dti"] == 999, "dti"] = np.nan
        return df

    train_df = fix_dti_sentinel(train_df)
    val_df = fix_dti_sentinel(val_df)
    test_df = fix_dti_sentinel(test_df)

    COLUMNS_TO_DROP = ["sec_app_earliest_cr_line"]
    train_df = train_df.drop(columns=[c for c in COLUMNS_TO_DROP if c in train_df.columns])
    val_df = val_df.drop(columns=[c for c in COLUMNS_TO_DROP if c in val_df.columns])
    test_df = test_df.drop(columns=[c for c in COLUMNS_TO_DROP if c in test_df.columns])

    y_train, y_val, y_test = train_df[TARGET_COL].copy(), val_df[TARGET_COL].copy(), test_df[TARGET_COL].copy()
    X_train = train_df.drop(columns=[TARGET_COL])
    X_val = val_df.drop(columns=[TARGET_COL])
    X_test = test_df.drop(columns=[TARGET_COL])

    MISSING_THRESHOLD = 0.01
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    na_pct = X_train[numeric_cols].isna().mean()
    low_missing_num_cols = na_pct[na_pct < MISSING_THRESHOLD].index.tolist()
    high_missing_num_cols = na_pct[na_pct >= MISSING_THRESHOLD].index.tolist()

    numeric_low_missing_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("power_transform", PowerTransformer(method="yeo-johnson", standardize=True)),
    ])
    numeric_high_missing_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("power_transform", PowerTransformer(method="yeo-johnson", standardize=True)),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num_low_missing", numeric_low_missing_pipeline, low_missing_num_cols),
            ("num_high_missing", numeric_high_missing_pipeline, high_missing_num_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop", verbose_feature_names_out=True,
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    print(f"Shape -> Train: {X_train_processed.shape} | Val: {X_val_processed.shape} | Test: {X_test_processed.shape}")

    feature_names = preprocessor.get_feature_names_out()

    joblib.dump(preprocessor, os.path.join(BASE_DIR, "phase7_preprocessor.joblib"))
    joblib.dump(feature_names, os.path.join(BASE_DIR, "phase7_feature_names.joblib"))
    np.savez(
        os.path.join(BASE_DIR, "phase7_processed_data.npz"),
        X_train=X_train_processed if not hasattr(X_train_processed, "toarray") else X_train_processed.toarray(),
        X_val=X_val_processed if not hasattr(X_val_processed, "toarray") else X_val_processed.toarray(),
        X_test=X_test_processed if not hasattr(X_test_processed, "toarray") else X_test_processed.toarray(),
        y_train=y_train.values, y_val=y_val.values, y_test=y_test.values,
    )
    print("Phase 7 done.")


# ##############################################################################
# PHASE 8 — Baseline Models
# ##############################################################################
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix, classification_report,
)

def _evaluate_model(model, X, y, name="", threshold=0.5):
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    roc_auc = roc_auc_score(y, y_proba)
    pr_auc = average_precision_score(y, y_proba)
    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    print(f"{name}: ROC-AUC={roc_auc:.4f} PR-AUC={pr_auc:.4f} F1={f1:.4f} "
          f"Precision={precision:.4f} Recall={recall:.4f}")
    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "f1": f1,
            "precision": precision, "recall": recall, "confusion_matrix": cm.tolist()}

def run_phase8():
    data = np.load(os.path.join(BASE_DIR, "phase7_processed_data.npz"))
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    # X_test/y_test عمداً لود نمی‌شوند - طبق قانون طلایی

    log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, n_jobs=-1)
    log_reg.fit(X_train, y_train)
    lr_train_metrics = _evaluate_model(log_reg, X_train, y_train, "LR - TRAIN")
    lr_val_metrics = _evaluate_model(log_reg, X_val, y_val, "LR - VAL")

    rf = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=20,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_train_metrics = _evaluate_model(rf, X_train, y_train, "RF - TRAIN")
    rf_val_metrics = _evaluate_model(rf, X_val, y_val, "RF - VAL")

    print(f"LR gap: {lr_train_metrics['roc_auc']-lr_val_metrics['roc_auc']:.4f}")
    print(f"RF gap: {rf_train_metrics['roc_auc']-rf_val_metrics['roc_auc']:.4f}")

    joblib.dump(log_reg, os.path.join(BASE_DIR, "phase8_logistic_regression.joblib"))
    joblib.dump(rf, os.path.join(BASE_DIR, "phase8_random_forest.joblib"))
    print("Phase 8 done.")


# ##############################################################################
# PHASE 9 — Hyperparameter Tuning
# ##############################################################################
"""
⚠️ شناخته‌شده: penalty صریحاً روی 'elasticnet' تنظیم نشده، پس l1_ratio در جستجوی
LR عملاً بی‌اثر است (فقط C واقعاً تیون می‌شود). کد دقیقاً همانی که اجرا شده
حفظ شده - برای اجرای درست‌تر در آینده باید penalty="elasticnet" اضافه شود.
"""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import uniform, randint

def run_phase9():
    RANDOM_STATE = 42
    N_ITER = 25
    CV_FOLDS = 3

    data = np.load(os.path.join(BASE_DIR, "phase7_processed_data.npz"))
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # --- Logistic Regression ---
    lr_param_dist = {"C": uniform(0.001, 10), "l1_ratio": uniform(0, 1)}
    lr_base = LogisticRegression(class_weight="balanced", max_iter=2000,
                                  random_state=RANDOM_STATE, solver="saga")
    lr_search = RandomizedSearchCV(lr_base, lr_param_dist, n_iter=N_ITER,
                                    scoring="average_precision", cv=cv,
                                    random_state=RANDOM_STATE, n_jobs=-1, verbose=1)
    lr_search.fit(X_train, y_train)
    best_lr = lr_search.best_estimator_
    print("Best LR params:", lr_search.best_params_)
    lr_val_metrics = _evaluate_model(best_lr, X_val, y_val, "Tuned LR - VAL")

    # --- Random Forest ---
    rf_param_dist = {
        "n_estimators": randint(100, 400), "max_depth": randint(3, 12),
        "min_samples_leaf": randint(20, 150), "max_features": ["sqrt", "log2", 0.3],
    }
    rf_base = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    rf_search = RandomizedSearchCV(rf_base, rf_param_dist, n_iter=N_ITER,
                                    scoring="average_precision", cv=cv,
                                    random_state=RANDOM_STATE, n_jobs=-1, verbose=1)
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    print("Best RF params:", rf_search.best_params_)
    rf_val_metrics = _evaluate_model(best_rf, X_val, y_val, "Tuned RF - VAL")

    # --- HistGradientBoosting ---
    hgb_param_dist = {
        "max_iter": randint(100, 400), "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.3), "min_samples_leaf": randint(20, 150),
        "l2_regularization": uniform(0.0, 1.0),
    }
    hgb_base = HistGradientBoostingClassifier(class_weight="balanced", random_state=RANDOM_STATE,
                                               early_stopping=True, validation_fraction=0.1,
                                               n_iter_no_change=15)
    hgb_search = RandomizedSearchCV(hgb_base, hgb_param_dist, n_iter=N_ITER,
                                     scoring="average_precision", cv=cv,
                                     random_state=RANDOM_STATE, n_jobs=-1, verbose=1)
    hgb_search.fit(X_train, y_train)
    best_hgb = hgb_search.best_estimator_
    print("Best HGB params:", hgb_search.best_params_)
    hgb_val_metrics = _evaluate_model(best_hgb, X_val, y_val, "Tuned HGB - VAL")

    joblib.dump(best_lr, os.path.join(BASE_DIR, "phase9_tuned_logistic_regression.joblib"))
    joblib.dump(best_rf, os.path.join(BASE_DIR, "phase9_tuned_random_forest.joblib"))
    joblib.dump(best_hgb, os.path.join(BASE_DIR, "phase9_tuned_histgradboost.joblib"))
    print("Phase 9 done. Final model chosen: Logistic Regression.")


# ##############################################################################
# PHASE 10 — Threshold Optimization  [PATH FIX: DATA_DIR was hardcoded to
#            D:/aashit/proccess -> normalized to BASE_DIR]
# ##############################################################################
from sklearn.metrics import precision_recall_curve

def run_phase10():
    data = np.load(os.path.join(BASE_DIR, "phase7_processed_data.npz"))
    X_val, y_val = data["X_val"], data["y_val"]
    model = joblib.load(os.path.join(BASE_DIR, "phase9_tuned_logistic_regression.joblib"))
    y_proba_val = model.predict_proba(X_val)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba_val)
    precisions_t, recalls_t = precisions[:-1], recalls[:-1]
    f1_scores = 2 * precisions_t * recalls_t / (precisions_t + recalls_t + 1e-10)
    f2_scores = 5 * precisions_t * recalls_t / (4 * precisions_t + recalls_t + 1e-10)

    best_f1_idx, best_f2_idx = np.argmax(f1_scores), np.argmax(f2_scores)
    print(f"F1-optimal threshold: {thresholds[best_f1_idx]:.4f}")
    print(f"F2-optimal threshold: {thresholds[best_f2_idx]:.4f}  <- انتخاب‌شده به عنوان 0.35")

    cost_scenarios = [
        ("محافظه‌کارانه (FN 3x)", 1, 3), ("متوسط (FN 5x)", 1, 5), ("سخت‌گیرانه (FN 10x)", 1, 10),
    ]
    for label, cost_fp, cost_fn in cost_scenarios:
        thr = cost_fp / (cost_fp + cost_fn)
        y_pred_thr = (y_proba_val >= thr).astype(int)
        print(f"{label}: threshold={thr:.4f} Precision={precision_score(y_val, y_pred_thr):.4f} "
              f"Recall={recall_score(y_val, y_pred_thr):.4f}")
    print("Phase 10 done. Chosen operating threshold: 0.35")


# ##############################################################################
# PHASE 11 — Model Interpretation  [PATH FIX: DATA_DIR was hardcoded to
#            D:/aashit/proccess -> normalized to BASE_DIR]
# ##############################################################################
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance

def run_phase11():
    data = np.load(os.path.join(BASE_DIR, "phase7_processed_data.npz"))
    X_val, y_val = data["X_val"], data["y_val"]
    feature_names = joblib.load(os.path.join(BASE_DIR, "phase7_feature_names.joblib"))
    model = joblib.load(os.path.join(BASE_DIR, "phase9_tuned_logistic_regression.joblib"))

    y_proba = model.predict_proba(X_val)[:, 1]
    y_pred_final = (y_proba >= FINAL_THRESHOLD).astype(int)
    print(classification_report(y_val, y_pred_final, target_names=["No-Default", "Default"]))

    fpr, tpr, _ = roc_curve(y_val, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"LR (AUC={roc_auc_score(y_val, y_proba):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "phase11_roc_curve.png"), dpi=120); plt.close()

    precisions, recalls, _ = precision_recall_curve(y_val, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, label=f"LR (AP={average_precision_score(y_val, y_proba):.3f})")
    plt.axhline(y=y_val.mean(), color="k", linestyle="--", alpha=0.4)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "phase11_pr_curve.png"), dpi=120); plt.close()

    prob_true, prob_pred = calibration_curve(y_val, y_proba, n_bins=10, strategy="quantile")
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, "o-")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("میانگین احتمال پیش‌بینی‌شده"); plt.ylabel("نسبت واقعی default")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "phase11_calibration_curve.png"), dpi=120); plt.close()

    coefs = model.coef_[0]
    odds_ratios = np.exp(coefs)
    coef_table = sorted(zip(feature_names, coefs, odds_ratios), key=lambda x: abs(x[1]), reverse=True)
    print("Top 10 coefficients:")
    for name, coef, odds in coef_table[:10]:
        print(f"  {name}: coef={coef:.4f} odds_ratio={odds:.4f}")

    perm_result = permutation_importance(model, X_val, y_val, scoring="average_precision",
                                          n_repeats=10, random_state=42, n_jobs=-1)
    perm_sorted_idx = perm_result.importances_mean.argsort()[::-1][:10]
    print("Top 10 permutation importance:")
    for idx in perm_sorted_idx:
        print(f"  {feature_names[idx]}: {perm_result.importances_mean[idx]:.5f}")

    print("Phase 11 done.")


# ##############################################################################
# CALIBRATION — Check (موجود) + Fix (TODO: اسکریپت فیت واقعی هنوز فرستاده نشده)
# ##############################################################################
from sklearn.metrics import brier_score_loss

def run_calibration_check():
    """
    این بخش فقط CHECK می‌کند (Brier score + reliability diagram روی val) -
    هیچ مدل جدیدی فیت نمی‌کند. مسیرهای اصلی این اسکریپت در نسخه‌ی دریافتی
    ناسازگار بودند (دو پوشه‌ی متفاوت برای preprocessor/val در برابر model) -
    اینجا با BASE_DIR یکدست شده؛ لطفاً مطمئن شو BASE_DIR شامل نسخه‌ی درست
    و هماهنگ هر سه فایل است.
    """
    preprocessor = joblib.load(os.path.join(BASE_DIR, "phase7_preprocessor.joblib"))
    model = joblib.load(os.path.join(BASE_DIR, "phase9_tuned_logistic_regression.joblib"))

    df = pd.read_csv(os.path.join(BASE_DIR, "split_val.csv"))
    y_true = df["target"].values
    X = df.drop(columns=["target"])
    if "dti" in X.columns:
        X.loc[X["dti"] == 999, "dti"] = np.nan

    X_transformed = preprocessor.transform(X)
    y_prob = model.predict_proba(X_transformed)[:, 1]

    brier = brier_score_loss(y_true, y_prob)
    print(f"Brier score: {brier:.4f}")

    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="quantile")
    for p, a in zip(mean_pred, frac_pos):
        print(f"  predicted ~{p:.3f}  ->  actual {a:.3f}  (diff {a - p:+.3f})")

    plt.figure(figsize=(6, 6))
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Actual fraction of defaults")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "calibration_curve.png"), dpi=150)
    print("Saved calibration_curve.png")


def run_calibration_fix_TODO():
    """
    ⚠️ TODO — اسکریپت واقعی این بخش هنوز در دسترس نیست.
    این باید شامل موارد زیر باشد (طبق چیزی که در README مستند شده):
      1. from sklearn.calibration import CalibratedClassifierCV
      2. یک LogisticRegression پایه (همان معماری Phase 9) را با
         CalibratedClassifierCV(method="sigmoid", cv=5) دوباره روی
         X_train/y_train فیت کند (فقط train، هرگز val/test).
      3. با probability های calibrated جدید، threshold را دوباره از روی
         validation مشتق کند (نتیجه‌ی مستند شده: ~0.10).
      4. مدل را با نام phase9_calibrated_logistic_regression.joblib ذخیره کند.
    لطفاً اگر این اسکریپت را داری، بفرست تا این تابع placeholder با کد واقعی
    جایگزین شود.
    """
    raise NotImplementedError(
        "اسکریپت واقعی calibration fix هنوز در اختیار نیست - به توضیحات docstring مراجعه کن."
    )


# ##############################################################################
# PHASE 13 — Final Test Evaluation (تنها و آخرین استفاده از split_test.csv)
# ##############################################################################
def run_phase13():
    TEST_PATH = os.path.join(BASE_DIR, "split_test.csv")
    COLUMNS_TO_DROP = ["sec_app_earliest_cr_line"]

    print("⚠️  بارگذاری split_test.csv - این تنها و آخرین استفاده از این فایل است")
    test_df = pd.read_csv(TEST_PATH)
    assert test_df.shape[1] == 90, "test باید 90 ستون داشته باشد!"

    if "dti" in test_df.columns:
        test_df.loc[test_df["dti"] == 999, "dti"] = np.nan
    test_df = test_df.drop(columns=[c for c in COLUMNS_TO_DROP if c in test_df.columns])

    y_test = test_df["target"].copy()
    X_test_raw = test_df.drop(columns=["target"])

    preprocessor = joblib.load(os.path.join(BASE_DIR, "phase7_preprocessor.joblib"))
    model = joblib.load(os.path.join(BASE_DIR, "phase9_tuned_logistic_regression.joblib"))

    X_test_processed = preprocessor.transform(X_test_raw)
    y_proba_test = model.predict_proba(X_test_processed)[:, 1]
    y_pred_test = (y_proba_test >= FINAL_THRESHOLD).astype(int)

    roc_auc = roc_auc_score(y_test, y_proba_test)
    pr_auc = average_precision_score(y_test, y_proba_test)
    print(f"TEST ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(classification_report(y_test, y_pred_test, target_names=["No-Default", "Default"]))

    VAL_ROC_AUC_FROM_PHASE9 = 0.7100
    VAL_PR_AUC_FROM_PHASE9 = 0.3751
    print(f"Val vs Test ROC-AUC diff: {roc_auc - VAL_ROC_AUC_FROM_PHASE9:.4f}")
    print(f"Val vs Test PR-AUC diff:  {pr_auc - VAL_PR_AUC_FROM_PHASE9:.4f}")
    print("Phase 13 done - این تنها ارزیابی 'تمیز' (single-look) روی test است.")
    print("⚠️  توجه: نسخه‌ی 'calibrated + threshold=0.10' که در README به عنوان")
    print("    نتیجه‌ی رسمی گزارش شده، بعداً و جدا از این اسکریپت، دوباره روی")
    print("    همین test ارزیابی شده - این دومین نگاه به test بوده، نه یک‌بار.")


# ##############################################################################
# MAIN — اجرای ترتیبی همه‌ی فازها
# ##############################################################################
if __name__ == "__main__":
    # هر خط را طبق نیاز کامنت/آن‌کامنت کن. اجرای کامل از ابتدا زمان‌بر است
    # (به‌خصوص Phase 9). فایل‌های میانی هر فاز باید از فاز قبل روی دیسک
    # موجود باشند (این فایل آن‌ها را دوباره نمی‌سازد در حافظه).

    # run_phase2()
    # run_phase3_sampling()
    # run_phase5()
    # run_phase6()
    # run_phase7()
    # run_phase8()
    # run_phase9()
    # run_phase10()
    # run_phase11()
    # run_calibration_check()
    # run_calibration_fix_TODO()   # <- هنوز پیاده نشده، اسکریپت واقعی لازم است
    # run_phase13()
    print("فازهای موردنظر را در بخش __main__ بالا آن‌کامنت کن و دوباره اجرا کن.")
