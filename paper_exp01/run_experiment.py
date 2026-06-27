"""
논문 재현 실험
  논문: 설명 가능한 정기예금 가입 여부 예측을 위한 앙상블 학습 기반 분류 모델들의 비교 분석
  데이터: paper_exp01/kaggle_data/bank/bank.csv (4,521행, y → deposit, sep=';')
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

import shap
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
import xgboost as xgb
import lightgbm as lgb

# ─── 한글 폰트 설정 (Windows) ───
for font in fm.findSystemFonts(fontpaths=None, fontext="ttf"):
    if any(k in font for k in ["malgun", "Malgun", "gulim", "Gulim"]):
        fm.fontManager.addfont(font)
        plt.rcParams["font.family"] = fm.FontProperties(fname=font).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False

# ─── 경로 ───
BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR / "kaggle_data" / "bank" / "bank.csv"
OUT_DIR    = BASE_DIR / "outputs"
METRIC_DIR = OUT_DIR / "metrics"
FIG_DIR    = OUT_DIR / "figures"
REP_DIR    = OUT_DIR / "reports"

RANDOM_STATE = 42

# ─── 논문 기준값 (Table 7, Table 8) ───
PAPER = {
    "with": {
        "RF":       {"accuracy": 0.857, "kappa": 0.715, "f1": 0.850},
        "GBM":      {"accuracy": 0.863, "kappa": 0.726, "f1": 0.860},
        "XGBoost":  {"accuracy": 0.855, "kappa": 0.710, "f1": 0.851},
        "LightGBM": {"accuracy": 0.864, "kappa": 0.727, "f1": 0.861},
    },
    "without": {
        "RF":       {"accuracy": 0.729, "kappa": 0.452, "f1": 0.694},
        "GBM":      {"accuracy": 0.738, "kappa": 0.469, "f1": 0.693},
        "XGBoost":  {"accuracy": 0.729, "kappa": 0.453, "f1": 0.690},
        "LightGBM": {"accuracy": 0.737, "kappa": 0.466, "f1": 0.690},
    },
}

# ══════════════════════════════════════════════════════════
# Step 1. 데이터 로드
# ══════════════════════════════════════════════════════════
print("=" * 60)
print("Step 1. 데이터 로드")
print("=" * 60)

df = pd.read_csv(DATA_PATH, sep=";")
df.columns = [c.strip().lower() for c in df.columns]
df.rename(columns={"y": "deposit"}, inplace=True)

print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\n결측치:\n{df.isnull().sum().to_string()}")
print(f"\ndeposit 분포:\n{df['deposit'].value_counts().to_string()}")

# ══════════════════════════════════════════════════════════
# Step 2. 전처리
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 2. 전처리")
print("=" * 60)

for col in ["default", "housing", "loan", "deposit"]:
    df[col] = df[col].map({"yes": 1, "no": 0})

y = df["deposit"].astype(int)
X = df.drop(columns=["deposit"])

X_enc = pd.get_dummies(X, drop_first=False)
print(f"원-핫 인코딩 후 독립변수 수: {X_enc.shape[1]}")

X_with    = X_enc.copy()
dur_cols  = [c for c in X_enc.columns if "duration" in c]
X_without = X_enc.drop(columns=dur_cols)

print(f"Dataset A (duration 포함): {X_with.shape}")
print(f"Dataset B (duration 제외): {X_without.shape}")

# ══════════════════════════════════════════════════════════
# Step 3. 모델 정의 (논문 Table 6 기반 고정 하이퍼파라미터)
# ══════════════════════════════════════════════════════════
def get_models():
    return {
        "RF": RandomForestClassifier(
            n_estimators=128,
            max_features="sqrt",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GBM": GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
        "XGBoost": xgb.XGBClassifier(
            booster="gbtree",
            n_estimators=500,
            max_depth=6,
            subsample=0.75,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            verbosity=0,
        ),
        "LightGBM": lgb.LGBMClassifier(
            boosting_type="gbdt",
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.5,
            colsample_bytree=1,
            random_state=RANDOM_STATE,
            verbose=-1,
        ),
    }

# ══════════════════════════════════════════════════════════
# Step 4. 10-Fold CV 평가
# ══════════════════════════════════════════════════════════
def run_cv(X, y, label):
    print(f"\n{'=' * 60}")
    print(f"Step 4. {label} - 10-Fold CV")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    models = get_models()
    rows = []

    for name, model in models.items():
        accs, kappas, f1s = [], [], []
        for tr, va in skf.split(X, y):
            Xtr, Xva = X.iloc[tr], X.iloc[va]
            ytr, yva = y.iloc[tr], y.iloc[va]
            model.fit(Xtr, ytr)
            yp = model.predict(Xva)
            accs.append(accuracy_score(yva, yp))
            kappas.append(cohen_kappa_score(yva, yp))
            # binary: F1 for positive class (deposit=1)
            f1s.append(f1_score(yva, yp, average="binary"))

        acc   = round(np.mean(accs),   3)
        kappa = round(np.mean(kappas), 3)
        f1    = round(np.mean(f1s),    3)
        print(f"  {name:10s}: Accuracy={acc:.3f}  Kappa={kappa:.3f}  F1={f1:.3f}")
        rows.append({"model": name, "accuracy": acc, "kappa": kappa, "f1": f1})

    return pd.DataFrame(rows), models

results_with,    models_with    = run_cv(X_with,    y, "Experiment 1: Duration 포함")
results_without, models_without = run_cv(X_without, y, "Experiment 2: Duration 제외")

# ══════════════════════════════════════════════════════════
# Step 5. 논문 결과 비교
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 5. 논문 결과 비교")
print("=" * 60)

def compare(res_df, paper_dict, label):
    rows = []
    for _, r in res_df.iterrows():
        m = r["model"]
        p = paper_dict[m]
        rows.append({
            "model":   m,
            "repr_acc":  r["accuracy"], "paper_acc":  p["accuracy"], "diff_acc":  round(r["accuracy"] - p["accuracy"], 3),
            "repr_kap":  r["kappa"],    "paper_kap":  p["kappa"],    "diff_kap":  round(r["kappa"]    - p["kappa"],    3),
            "repr_f1":   r["f1"],       "paper_f1":   p["f1"],       "diff_f1":   round(r["f1"]       - p["f1"],       3),
        })
    df_c = pd.DataFrame(rows)
    print(f"\n[{label}]")
    print(df_c.to_string(index=False))
    return df_c

comp_with    = compare(results_with,    PAPER["with"],    "Duration 포함")
comp_without = compare(results_without, PAPER["without"], "Duration 제외")

# ── CSV 저장 ──
results_with.to_csv(   METRIC_DIR / "duration_included_results.csv",    index=False, encoding="utf-8-sig")
results_without.to_csv(METRIC_DIR / "duration_excluded_results.csv",    index=False, encoding="utf-8-sig")
comp_with.to_csv(      METRIC_DIR / "paper_comparison_included.csv",    index=False, encoding="utf-8-sig")
comp_without.to_csv(   METRIC_DIR / "paper_comparison_excluded.csv",    index=False, encoding="utf-8-sig")
print("\n결과 CSV 저장 완료 →", METRIC_DIR)

# ══════════════════════════════════════════════════════════
# Step 6. SHAP 분석
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 6. SHAP 분석")
print("=" * 60)

def shap_analysis(X, y, model_name, models_dict, tag):
    model = models_dict[model_name]
    model.fit(X, y)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # binary classification: shap_values may be list [cls0, cls1] or ndarray
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Summary plot
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, X, show=False, max_display=15)
    plt.title(f"SHAP Summary — {tag} ({model_name})", pad=12)
    plt.tight_layout()
    fname = FIG_DIR / f"shap_summary_{tag}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  저장: {fname.name}")

    # Dependence plots for key variables
    key_vars = ["duration", "age", "balance", "housing_yes", "poutcome_success"]
    col_names = list(X.columns)
    for var in key_vars:
        if var not in col_names:
            continue
        try:
            col_idx = col_names.index(var)
            feat_vals  = X[var].values
            shap_vals  = sv[:, col_idx] if sv.ndim == 2 else sv[:, col_idx, 1]

            fig, ax = plt.subplots(figsize=(8, 5))
            sc = ax.scatter(feat_vals, shap_vals, c=shap_vals,
                            cmap="coolwarm", alpha=0.4, s=12, linewidths=0)
            plt.colorbar(sc, ax=ax, label=f"SHAP ({var})")
            ax.axhline(0, color="gray", linewidth=0.7, linestyle="--")
            ax.set_xlabel(var)
            ax.set_ylabel(f"SHAP value for\n{var}")
            ax.set_title(f"SHAP Dependence: {var} ({tag})")
            ax.grid(alpha=0.2)
            plt.tight_layout()
            fname_d = FIG_DIR / f"shap_dep_{var}_{tag}.png"
            plt.savefig(fname_d, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  저장: {fname_d.name}")
        except Exception as e:
            plt.close()
            print(f"  건너뜀 ({var}): {e}")

best_with    = results_with.loc[   results_with["accuracy"].idxmax(),    "model"]
best_without = results_without.loc[results_without["accuracy"].idxmax(), "model"]

print(f"\n[Duration 포함] 최고 모델: {best_with}")
shap_analysis(X_with, y, best_with, models_with, "duration_included")

print(f"\n[Duration 제외] 최고 모델: {best_without}")
shap_analysis(X_without, y, best_without, models_without, "duration_excluded")

# 모델 성능 비교 막대 그래프
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
model_names = results_with["model"].tolist()
x = np.arange(len(model_names))
w = 0.35

for ax, metric, mlabel in zip(axes,
                               ["accuracy", "kappa", "f1"],
                               ["Accuracy", "Cohen's Kappa", "F1-Score"]):
    ax.bar(x - w/2, results_with[metric],    w, label="Duration 포함", color="steelblue")
    ax.bar(x + w/2, results_without[metric], w, label="Duration 제외", color="coral")
    ax.set_title(mlabel)
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle("Model Performance Comparison (Duration Included vs Excluded)", fontsize=12)
plt.tight_layout()
perf_fig = FIG_DIR / "model_performance_comparison.png"
plt.savefig(perf_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  저장: {perf_fig.name}")

# ══════════════════════════════════════════════════════════
# Step 7. 보고서 생성
# ══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("Step 7. 보고서 생성")
print("=" * 60)

def df_to_md(df):
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows   = "\n".join(
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    )
    return "\n".join([header, sep, rows])

report = f"""# 논문 재현 실험 보고서

## 1. 실험 개요

| 항목 | 내용 |
|------|------|
| 논문 | 설명 가능한 정기예금 가입 여부 예측을 위한 앙상블 학습 기반 분류 모델들의 비교 분석 |
| 데이터 | bank.csv (4,521행 / sep=';' / y → deposit) |
| 모델 | RF, GBM, XGBoost, LightGBM |
| 평가 방식 | StratifiedKFold(n_splits=10, shuffle=True, random_state=42) |
| 평가 지표 | Accuracy, Cohen's Kappa, F1-Score (average='binary') |

## 2. 전처리

- `y` → `deposit` 컬럼명 변경
- 이진 변수 (default, housing, loan, deposit): yes=1, no=0
- 문자형 변수: One-Hot Encoding (drop_first=False)
- Dataset A (duration 포함): {X_with.shape[1]}개 독립변수
- Dataset B (duration 제외): {X_without.shape[1]}개 독립변수

## 3. 모델 하이퍼파라미터 (논문 Table 6 기반 고정값)

| 모델 | 설정 |
|------|------|
| RF | n_estimators=128, max_features='sqrt' |
| GBM | n_estimators=500, learning_rate=0.05, max_depth=5 |
| XGBoost | booster='gbtree', n_estimators=500, max_depth=6, subsample=0.75, eval_metric='logloss' |
| LightGBM | boosting_type='gbdt', n_estimators=1000, learning_rate=0.05, num_leaves=64, subsample=0.5 |

## 4. 실험 결과

### Experiment 1: Duration 포함

{df_to_md(results_with)}

### Experiment 2: Duration 제외

{df_to_md(results_without)}

## 5. 논문 결과 비교

### Duration 포함

{df_to_md(comp_with)}

### Duration 제외

{df_to_md(comp_without)}

## 6. SHAP 분석

- Duration 포함 최고 모델: **{best_with}**
- Duration 제외 최고 모델: **{best_without}**
- 생성 파일: `outputs/figures/` 참조

## 7. 재현 한계

1. **데이터 규모**: bank.csv(4,521행) vs 논문 Kaggle 버전(11,163행) — 수치 차이의 주 원인
2. **라이브러리 버전**: Python 3.12 / scikit-learn 1.x vs 논문 Python 3.7.9 / scikit-learn 0.24.2
3. **하이퍼파라미터**: 논문 Table 6의 GridSearchCV 최종 선택값 미확인 — 합리적 기본값 사용
"""

rep_path = REP_DIR / "reproduction_summary.md"
rep_path.write_text(report, encoding="utf-8")
print(f"보고서 저장 → {rep_path}")

print("\n" + "=" * 60)
print("실험 완료")
print(f"  metrics : {METRIC_DIR}")
print(f"  figures : {FIG_DIR}")
print(f"  reports : {REP_DIR}")
print("=" * 60)
