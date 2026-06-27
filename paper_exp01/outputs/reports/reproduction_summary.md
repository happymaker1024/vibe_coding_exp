# 논문 재현 실험 보고서

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
- Dataset A (duration 포함): 48개 독립변수
- Dataset B (duration 제외): 47개 독립변수

## 3. 모델 하이퍼파라미터 (논문 Table 6 기반 고정값)

| 모델 | 설정 |
|------|------|
| RF | n_estimators=128, max_features='sqrt' |
| GBM | n_estimators=500, learning_rate=0.05, max_depth=5 |
| XGBoost | booster='gbtree', n_estimators=500, max_depth=6, subsample=0.75, eval_metric='logloss' |
| LightGBM | boosting_type='gbdt', n_estimators=1000, learning_rate=0.05, num_leaves=64, subsample=0.5 |

## 4. 실험 결과

### Experiment 1: Duration 포함

| model | accuracy | kappa | f1 |
| --- | --- | --- | --- |
| RF | 0.898 | 0.304 | 0.346 |
| GBM | 0.902 | 0.452 | 0.504 |
| XGBoost | 0.898 | 0.428 | 0.482 |
| LightGBM | 0.903 | 0.451 | 0.502 |

### Experiment 2: Duration 제외

| model | accuracy | kappa | f1 |
| --- | --- | --- | --- |
| RF | 0.888 | 0.176 | 0.212 |
| GBM | 0.882 | 0.225 | 0.278 |
| XGBoost | 0.874 | 0.216 | 0.277 |
| LightGBM | 0.883 | 0.222 | 0.273 |

## 5. 논문 결과 비교

### Duration 포함

| model | repr_acc | paper_acc | diff_acc | repr_kap | paper_kap | diff_kap | repr_f1 | paper_f1 | diff_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RF | 0.898 | 0.857 | 0.041 | 0.304 | 0.715 | -0.411 | 0.346 | 0.85 | -0.504 |
| GBM | 0.902 | 0.863 | 0.039 | 0.452 | 0.726 | -0.274 | 0.504 | 0.86 | -0.356 |
| XGBoost | 0.898 | 0.855 | 0.043 | 0.428 | 0.71 | -0.282 | 0.482 | 0.851 | -0.369 |
| LightGBM | 0.903 | 0.864 | 0.039 | 0.451 | 0.727 | -0.276 | 0.502 | 0.861 | -0.359 |

### Duration 제외

| model | repr_acc | paper_acc | diff_acc | repr_kap | paper_kap | diff_kap | repr_f1 | paper_f1 | diff_f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RF | 0.888 | 0.729 | 0.159 | 0.176 | 0.452 | -0.276 | 0.212 | 0.694 | -0.482 |
| GBM | 0.882 | 0.738 | 0.144 | 0.225 | 0.469 | -0.244 | 0.278 | 0.693 | -0.415 |
| XGBoost | 0.874 | 0.729 | 0.145 | 0.216 | 0.453 | -0.237 | 0.277 | 0.69 | -0.413 |
| LightGBM | 0.883 | 0.737 | 0.146 | 0.222 | 0.466 | -0.244 | 0.273 | 0.69 | -0.417 |

## 6. SHAP 분석

- Duration 포함 최고 모델: **LightGBM**
- Duration 제외 최고 모델: **RF**
- 생성 파일: `outputs/figures/` 참조

## 7. 재현 한계

1. **데이터 규모**: bank.csv(4,521행) vs 논문 Kaggle 버전(11,163행) — 수치 차이의 주 원인
2. **라이브러리 버전**: Python 3.12 / scikit-learn 1.x vs 논문 Python 3.7.9 / scikit-learn 0.24.2
3. **하이퍼파라미터**: 논문 Table 6의 GridSearchCV 최종 선택값 미확인 — 합리적 기본값 사용
