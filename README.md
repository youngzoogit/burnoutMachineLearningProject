# 직장인 번아웃 위험도 예측 프로젝트

## 1. 프로젝트 개요

본 프로젝트는 직장인의 근무환경, 스트레스 수준, 워라밸, 직무 만족도, 생산성 등의 데이터를 활용하여 `burnout_risk_score`를 예측하는 머신러닝 회귀 프로젝트입니다.

초기에는 이직 의향 예측도 함께 시도하였으나, 예측 성능이 낮아 최종 분석에서는 번아웃 위험 점수 예측에 집중하였습니다.

## 2. 사용 데이터

- 데이터 출처: Kaggle 공개 데이터셋
- 데이터 주제: 직장인 정신건강 및 근무환경 데이터
- 주요 변수:
  - 근무환경: `weekly_work_hours`, `weekly_overtime_hours`, `work_model`
  - 심리·조직 요인: `stress_level`, `work_life_balance_score`, `job_satisfaction_score`
  - 성과 및 상태: `productivity_score`, `absenteeism_days_per_year`
  - 예측 대상: `burnout_risk_score`

## 3. 분석 과정

1. 데이터 확인 및 결측치 처리
2. 범주형 변수 원-핫 인코딩
3. 데이터 누수 가능성이 있는 변수 제거
4. 수치형 변수 상관관계 분석
5. 회귀 모델 학습 및 성능 비교
6. Feature Importance 및 SHAP 분석
7. Streamlit 기반 예측 화면 구현

## 4. 사용 모델

다음 회귀 모델을 비교하였습니다.

- Linear Regression
- Elastic Net
- RandomForestRegressor
- XGBRegressor

## 5. 모델 성능 비교

| 모델 | R² | RMSE | MAE |
|---|---:|---:|---:|
| Linear Regression | 0.78 | 1.06 | 0.80 |
| Elastic Net | 0.78 | 1.07 | 0.81 |
| RandomForestRegressor | 0.80 | 1.01 | 0.76 |
| XGBRegressor | 0.82 | 0.97 | 0.74 |

XGBRegressor가 가장 낮은 오차와 가장 높은 설명력을 보여 최종 모델로 선정하였습니다.

## 6. 주요 분석 결과

SHAP 분석 결과, 번아웃 위험 점수에 주요하게 영향을 미친 변수는 다음과 같습니다.

- `weekly_overtime_hours`: 초과근무 시간이 많을수록 번아웃 위험 증가
- `work_life_balance_score`: 워라밸 점수가 높을수록 번아웃 위험 감소
- `absenteeism_days_per_year`: 결근일수가 많을수록 번아웃 위험 증가
- `stress_level`: 높은 스트레스 수준은 번아웃 위험 증가에 영향

이를 통해 번아웃은 단순히 근무시간만의 문제가 아니라, 근무환경과 심리적 요인이 복합적으로 작용한 결과임을 확인하였습니다.

## 7. Streamlit 실행 방법

### 1) 가상환경 생성 및 활성화

```bash
python -m venv .venv
```

Windows PowerShell 기준:

```bash
.\.venv\Scripts\activate
```

### 2) 패키지 설치

```bash
pip install -r requirements.txt
```

### 3) Streamlit 앱 실행

```bash
streamlit run burnout_app.py
```

## 8. 프로젝트 폴더 구조 예시

```text
project/
├── README.md
├── requirements.txt
├── burnout_app.py
├── mlProject_Regression.ipynb
├── dataset/
│   └── mental_health_workforce_synthetic.csv
└── model/
    ├── burnout_model.joblib
    ├── burnout_encoder.joblib
    ├── burnout_scaler.joblib
    └── burnout_features_meta.joblib
```

## 9. 트러블슈팅

### 이직 의향 예측 모델 성능 저하

초기에는 `intention_to_leave`를 종속변수로 설정하여 이직 의향 예측을 시도하였습니다. 그러나 예측 성능이 낮고, 특히 이직 의향이 있는 집단을 충분히 예측하지 못하는 문제가 확인되었습니다. SMOTE를 적용하여 데이터 불균형을 보완하였으나 성능 개선 효과가 크지 않아 최종 분석에서는 제외하였습니다.

### 데이터 누수 방지

`mental_health_condition`, `has_diagnosis`, `treatment_type`은 정신건강 진단 및 치료와 관련된 결과성 변수이므로, 모델 성능이 과대평가될 가능성이 있다고 판단하여 독립변수에서 제거하였습니다.

### 다중공선성 처리

`weekly_work_hours`와 `weekly_overtime_hours`의 상관계수가 0.95로 높게 나타나, 번아웃과 더 직접적으로 관련된 `weekly_overtime_hours`를 유지하고 `weekly_work_hours`는 모델 학습에서 제외하였습니다.

## 10. 한계점 및 개선 방향

본 모델은 번아웃 위험 점수를 예측하는 참고 지표이며, 실제 정신건강 진단을 대체할 수는 없습니다. 향후 실제 기업의 근태, 업무량, 조직문화, 상담 이력 등의 데이터를 추가로 확보한다면 모델의 현실성과 신뢰도를 높일 수 있습니다.

또한 SHAP 분석 결과를 Streamlit 화면에 연동하면, 단순 점수 예측을 넘어 개인별 위험 요인과 개선 방향을 함께 제공하는 서비스로 확장할 수 있습니다.

## 11. 기술 스택

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Joblib
