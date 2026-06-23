import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# Streamlit Cloud / Linux 환경의 나눔고딕 폰트 경로
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "NanumGothic"
else:
    # 로컬 Windows용
    plt.rcParams["font.family"] = "Malgun Gothic"

# 마이너스 기호 깨짐 방지
plt.rcParams["axes.unicode_minus"] = False

# 0. 페이지 기본 설정

st.set_page_config(
    page_title="직장인 번아웃 위험도 예측 시스템",
    layout="wide"
)

# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# 1. 모델, 인코더, 스케일러 로드

@st.cache_resource
def load_ml_objects():
    encoder = joblib.load("model/encoder.joblib")
    scaler = joblib.load("model/scaler.joblib")
    model = joblib.load("model/model.joblib")
    meta = joblib.load("model/features_meta.joblib")
    return encoder, scaler, model, meta


try:
    encoder, scaler, model, meta = load_ml_objects()

    categorical_cols = meta["categorical_cols"]

    # ----------------------------------------------------------------
    # feature_cols 복원:
    #   meta에 feature_cols 키가 저장되지 않은 경우,
    #   scaler.feature_names_in_ 를 정답 기준으로 사용합니다.
    #   (scaler가 학습 당시 본 컬럼 순서와 정확히 일치)
    # ----------------------------------------------------------------
    if "feature_cols" in meta:
        feature_cols = meta["feature_cols"]
    else:
        feature_cols = list(scaler.feature_names_in_)

    # 인코딩된 범주형 컬럼명 추출
    encoded_cat_cols = encoder.get_feature_names_out(categorical_cols)

    # scaler가 기대하는 수치형 컬럼 = feature_cols 중 인코딩 컬럼이 아닌 것
    encoded_cat_set = set(encoded_cat_cols)
    numeric_cols = [c for c in feature_cols if c not in encoded_cat_set]

except FileNotFoundError:
    st.error("모델 파일을 찾을 수 없습니다. 먼저 주피터 노트북에서 모델을 학습하고 joblib 파일을 저장해주세요.")
    st.stop()


# 2. 원본 데이터 로드

@st.cache_data
def load_raw_data():
    # 실제 저장된 파일명: mental_health_workplace.csv
    df = pd.read_csv("dataset/mental_health_workplace.csv")

    # 결측치 처리
    if "mental_health_condition" in df.columns:
        df["mental_health_condition"] = df["mental_health_condition"].fillna("Healthy")

    if "treatment_type" in df.columns:
        df["treatment_type"] = df["treatment_type"].fillna("Not Applicable")

    if "used_eap" in df.columns:
        df["used_eap"] = df["used_eap"].fillna("Not Available")

    return df


df = load_raw_data()


# 3. 보조 함수

def get_burnout_group(score):
    """
    예측된 burnout_risk_score를 Low / Medium / High로 변환
    """
    if score <= 3:
        return "Low"
    elif score <= 6:
        return "Medium"
    else:
        return "High"


def get_group_message(group):
    """
    번아웃 그룹별 안내 메시지
    """
    if group == "Low":
        return "낮은 번아웃 위험군입니다. 현재 상태는 비교적 안정적인 편입니다."
    elif group == "Medium":
        return "중간 수준의 번아웃 위험군입니다. 근무시간, 스트레스, 워라밸 관리가 필요합니다."
    else:
        return "높은 번아웃 위험군입니다. 초과근무, 스트레스, 직무 만족도에 대한 적극적인 관리가 필요합니다."


def make_input_dataframe(input_dict):
    """
    사용자 입력값을 모델 입력 형태로 변환
    """
    input_df = pd.DataFrame([input_dict])

    # 범주형 인코딩
    input_cat = encoder.transform(input_df[categorical_cols])

    encoded_cat_cols = encoder.get_feature_names_out(categorical_cols)

    input_cat_df = pd.DataFrame(
        input_cat,
        columns=encoded_cat_cols
    )

    # 수치형 + 범주형 결합
    input_encoded = pd.concat(
        [
            input_df[numeric_cols].reset_index(drop=True),
            input_cat_df.reset_index(drop=True)
        ],
        axis=1
    )

    # 학습 당시 컬럼 순서와 동일하게 정렬
    input_encoded = input_encoded.reindex(
        columns=feature_cols,
        fill_value=0
    )

    # 스케일링
    input_scaled = scaler.transform(input_encoded)

    return input_encoded, input_scaled


# 4. 제목

st.title("🔥직장인 번아웃 위험도 예측")
st.write(
    """
    직원의 근무환경, 직무 만족도, 스트레스, 생활습관 정보를 입력하면  
    머신러닝 모델이 예상 번아웃 위험 점수를 예측합니다.
    """
)

main_tab1, main_tab2 = st.tabs(
    ["번아웃 위험도 시뮬레이터", "데이터 인사이트 대시보드"]
)


# TAB 1. 번아웃 위험도 시뮬레이터

with main_tab1:
    st.markdown("## 개별 직원 번아웃 위험도 예측")
    st.write("아래 정보를 입력한 뒤 예측 버튼을 눌러주세요.")

    col1, col2 = st.columns([2, 1.1])

    # 입력 영역
    with col1:
        st.subheader("직원 정보 입력")

        with st.container(border=True):

            input_dict = {}

            # 범주형 컬럼명 → 한글 라벨 매핑
            CAT_LABEL = {
                "country":                    "국가",
                "industry":                   "산업군",
                "job_role":                   "직무",
                "employment_type":            "고용 형태",
                "work_model":                 "근무 방식",
                "company_size":               "회사 규모",
                "age_group":                  "연령대",
                "gender":                     "성별",
                "stress_level":               "스트레스 수준",
                "employer_support_level":     "고용주 지원 수준",
                "mental_health_policy_exists": "정신건강 정책 유무",
                "eap_available":              "EAP(직원지원프로그램) 제공 여부",
                "used_eap":                   "EAP 이용 여부",
                "workplace_stigma_felt":      "직장 내 낙인감",
                "remote_work_preference":     "원격근무 선호도",
            }

            # 범주형 옵션값 → 한글 매핑 (컬럼별)
            CAT_OPTIONS_KO = {
                "country": {
                    "Australia": "호주", "Brazil": "브라질", "Canada": "캐나다",
                    "Denmark": "덴마크", "France": "프랑스", "Germany": "독일",
                    "India": "인도", "Ireland": "아일랜드", "Japan": "일본",
                    "Netherlands": "네덜란드", "New Zealand": "뉴질랜드",
                    "Norway": "노르웨이", "Pakistan": "파키스탄",
                    "Singapore": "싱가포르", "South Africa": "남아프리카공화국",
                    "South Korea": "대한민국", "Sweden": "스웨덴",
                    "UAE": "아랍에미리트", "UK": "영국", "USA": "미국",
                },
                "industry": {
                    "Construction": "건설", "Consulting": "컨설팅",
                    "Education": "교육", "Finance & Banking": "금융·은행",
                    "Government & Public Sector": "정부·공공기관",
                    "Healthcare": "의료·헬스케어", "Hospitality": "숙박·관광",
                    "Legal": "법률", "Manufacturing": "제조",
                    "Media & Entertainment": "미디어·엔터테인먼트",
                    "Nonprofit": "비영리", "Pharmaceuticals": "제약",
                    "Retail & E-commerce": "유통·이커머스", "Technology": "IT·기술",
                },
                "job_role": {
                    "Accountant": "회계사", "Consultant": "컨설턴트",
                    "Customer Service": "고객 서비스", "Data Scientist": "데이터 사이언티스트",
                    "Designer": "디자이너", "Doctor/Physician": "의사",
                    "Executive/C-Suite": "임원·C레벨", "Financial Analyst": "재무 분석가",
                    "HR Professional": "인사 담당자", "Lawyer": "변호사",
                    "Manager/Team Lead": "관리자·팀장",
                    "Marketing Specialist": "마케팅 전문가", "Nurse": "간호사",
                    "Operations Manager": "운영 관리자", "Product Manager": "프로덕트 매니저",
                    "Research Scientist": "연구 과학자",
                    "Sales Representative": "영업 담당자", "Social Worker": "사회복지사",
                    "Software Engineer": "소프트웨어 엔지니어", "Teacher": "교사",
                },
                "employment_type": {
                    "Contract": "계약직", "Freelance": "프리랜서",
                    "Full-time": "정규직", "Part-time": "파트타임",
                },
                "work_model": {
                    "Hybrid": "하이브리드", "On-site": "현장 근무", "Remote": "원격 근무",
                },
                "company_size": {
                    "Startup (1-50)": "스타트업 (1~50명)",
                    "Small (51-200)": "소기업 (51~200명)",
                    "Medium (201-1000)": "중기업 (201~1,000명)",
                    "Large (1001-5000)": "대기업 (1,001~5,000명)",
                    "Enterprise (5000+)": "초대기업 (5,000명+)",
                },
                "age_group": {
                    "18-24": "18~24세", "25-34": "25~34세", "35-44": "35~44세",
                    "45-54": "45~54세", "55-64": "55~64세", "65+": "65세 이상",
                },
                "gender": {
                    "Female": "여성", "Male": "남성",
                    "Non-binary": "논바이너리", "Prefer not to say": "밝히지 않음",
                },
                "stress_level": {
                    "Very Low": "매우 낮음", "Low": "낮음", "Moderate": "보통",
                    "High": "높음", "Very Severe": "매우 심각",
                },
                "employer_support_level": {
                    "Poor": "미흡", "Average": "보통", "Good": "좋음", "Excellent": "매우 좋음",
                },
                "mental_health_policy_exists": {
                    "No": "없음", "Partial": "일부 있음", "Yes": "있음",
                },
                "eap_available": {
                    "No": "미제공", "Yes": "제공",
                },
                "used_eap": {
                    "No": "이용 안 함", "Yes": "이용함",
                    "Not Available": "프로그램 없음",
                },
                "workplace_stigma_felt": {
                    "None": "없음", "Mild": "약간",
                    "Moderate": "보통", "Severe": "심각",
                },
                "remote_work_preference": {
                    "No Preference": "무관",
                    "Prefer On-site": "현장 근무 선호",
                    "Prefer Remote": "원격 근무 선호",
                    "Strongly Prefer On-site": "현장 근무 강하게 선호",
                    "Strongly Prefer Remote": "원격 근무 강하게 선호",
                },
            }

            st.markdown("### 기본 정보")

            sub1, sub2 = st.columns(2)

            with sub1:
                if "years_of_experience" in numeric_cols:
                    input_dict["years_of_experience"] = st.number_input(
                        "경력 연수",
                        min_value=0,
                        max_value=50,
                        value=5
                    )

            with sub2:
                if "annual_salary_usd" in numeric_cols:
                    input_dict["annual_salary_usd"] = st.number_input(
                        "연봉(USD)",
                        min_value=0,
                        max_value=300000,
                        value=60000,
                        step=1000
                    )

            st.markdown("### 근무 환경")

            sub4, sub5 = st.columns(2)

            with sub4:
                if "weekly_overtime_hours" in numeric_cols:
                    input_dict["weekly_overtime_hours"] = st.slider(
                        "주간 초과근무 시간",
                        min_value=0,
                        max_value=40,
                        value=5
                    )

            with sub5:
                if "absenteeism_days_per_year" in numeric_cols:
                    input_dict["absenteeism_days_per_year"] = st.slider(
                        "연간 결근일수",
                        min_value=0,
                        max_value=60,
                        value=5
                    )

            st.markdown("### 직무 및 생활 지표")

            sub6, sub7, sub8 = st.columns(3)

            with sub6:
                if "work_life_balance_score" in numeric_cols:
                    input_dict["work_life_balance_score"] = st.slider(
                        "워라밸 점수",
                        min_value=1.0,
                        max_value=10.0,
                        value=6.0,
                        step=0.1
                    )

            with sub7:
                if "job_satisfaction_score" in numeric_cols:
                    input_dict["job_satisfaction_score"] = st.slider(
                        "직무 만족도 점수",
                        min_value=1.0,
                        max_value=10.0,
                        value=6.0,
                        step=0.1
                    )

            with sub8:
                if "productivity_score" in numeric_cols:
                    input_dict["productivity_score"] = st.slider(
                        "생산성 점수",
                        min_value=1.0,
                        max_value=10.0,
                        value=6.0,
                        step=0.1
                    )

            sub9, sub10, sub11 = st.columns(3)

            with sub9:
                if "manager_support_score" in numeric_cols:
                    input_dict["manager_support_score"] = st.slider(
                        "관리자 지원 점수",
                        min_value=1.0,
                        max_value=10.0,
                        value=6.0,
                        step=0.1
                    )

            with sub10:
                if "team_collaboration_score" in numeric_cols:
                    input_dict["team_collaboration_score"] = st.slider(
                        "팀 협업 점수",
                        min_value=1.0,
                        max_value=10.0,
                        value=6.0,
                        step=0.1
                    )

            with sub11:
                if "sleep_hours_per_night" in numeric_cols:
                    input_dict["sleep_hours_per_night"] = st.slider(
                        "하루 평균 수면시간",
                        min_value=0.0,
                        max_value=12.0,
                        value=7.0,
                        step=0.1
                    )

            if "exercise_days_per_week" in numeric_cols:
                input_dict["exercise_days_per_week"] = st.slider(
                    "주간 운동 일수",
                    min_value=0,
                    max_value=7,
                    value=2
                )

            st.markdown("### 기타 정보")

            # 범주형 selectbox: 한글로 표시하되 실제 영어 원본값을 input_dict에 저장
            for col in categorical_cols:
                raw_options = sorted(df[col].dropna().unique())

                if len(raw_options) == 0:
                    raw_options = ["Unknown"]

                # 해당 컬럼의 한글 옵션 매핑 (없으면 원본 그대로)
                ko_map = CAT_OPTIONS_KO.get(col, {})
                ko_options = [ko_map.get(v, v) for v in raw_options]

                # 한글 라벨로 표시
                label = CAT_LABEL.get(col, col)
                selected_ko = st.selectbox(label, options=ko_options)

                # 선택된 한글값 → 영어 원본값 역변환 후 모델 입력에 사용
                ko_to_en = {v: k for k, v in ko_map.items()}
                input_dict[col] = ko_to_en.get(selected_ko, selected_ko)

            # numeric_cols 중 위에서 입력하지 않은 컬럼은 중앙값으로 자동 채움
            # (year 포함 — 연도는 UI에서 제거했으나 모델 입력에는 중앙값 자동 반영)
            for col in numeric_cols:
                if col not in input_dict:
                    input_dict[col] = float(df[col].median())

    # ------------------------------------------------------------
    # 예측 결과 영역
    # ------------------------------------------------------------
    with col2:
        st.subheader("예측 결과")

        with st.container(border=True):

            if st.button("번아웃 위험도 예측 실행", use_container_width=True, type="primary"):

                input_encoded, input_scaled = make_input_dataframe(input_dict)

                pred_score = model.predict(input_scaled)[0]

                # 예측값이 1~10 범위를 살짝 벗어날 수 있으므로 표시용으로 보정
                pred_score_clipped = np.clip(pred_score, 1, 10)

                burnout_group = get_burnout_group(pred_score_clipped)

                st.metric(
                    label="예상 번아웃 위험 점수",
                    value=f"{pred_score_clipped:.2f} / 10"
                )

                st.markdown("---")

                if burnout_group == "Low":
                    st.success(f"위험군: {burnout_group}")
                elif burnout_group == "Medium":
                    st.warning(f"위험군: {burnout_group}")
                else:
                    st.error(f"위험군: {burnout_group}")

                st.write(get_group_message(burnout_group))

                st.markdown("---")

                # 원본 데이터 기준 해당 그룹의 정신건강 진단 비율 표시
                df_temp = df.copy()
                df_temp["burnout_group"] = pd.cut(
                    df_temp["burnout_risk_score"],
                    bins=[0, 3, 6, 10],
                    labels=["Low", "Medium", "High"],
                    include_lowest=True
                )

                diagnosis_rate = pd.crosstab(
                    df_temp["burnout_group"],
                    df_temp["has_diagnosis"],
                    normalize="index"
                ) * 100

                if "Yes" in diagnosis_rate.columns:
                    group_yes_rate = diagnosis_rate.loc[burnout_group, "Yes"]

                    st.info(
                        f"데이터 분석 기준, {burnout_group} 그룹의 정신건강 진단 경험 비율은 "
                        f"약 {group_yes_rate:.1f}%입니다."
                    )

            else:
                st.info("왼쪽 입력값을 설정한 뒤 예측 버튼을 눌러주세요.")


# TAB 2. 데이터 인사이트 대시보드

with main_tab2:
    st.markdown("## 데이터 인사이트 대시보드")
    st.write(
        "10,000명 직장인의 직무 환경·정신건강·번아웃 데이터를 다각도로 분석한 결과입니다. "
        "각 탭을 클릭하여 세부 분석을 확인하세요."
    )

    # 데이터 전처리: 번아웃 그룹 컬럼 추가 (대시보드 전체 공유)
    df_dash = df.copy()
    df_dash["burnout_group"] = pd.cut(
        df_dash["burnout_risk_score"],
        bins=[0, 3, 6, 10],
        labels=["Low", "Medium", "High"],
        include_lowest=True
    )

    (
        sub_tab1, sub_tab2, sub_tab3,
        sub_tab4, sub_tab5, sub_tab6, sub_tab7
    ) = st.tabs([
        "📊 데이터 개요",
        "🔥 번아웃 분포",
        "🔗 변수 상관관계",
        "🏆 모델 성능 비교",
        "🎯 피처 중요도",
        "🏥 정신건강 진단",
        "🔍 주요 변수 분석",
    ])

    # ============================================================
    # Sub Tab 1. 데이터 개요 (EDA)
    # ============================================================
    with sub_tab1:
        st.subheader("📊 데이터셋 기본 정보")
        st.markdown(
            """
            > 본 분석에 사용된 데이터셋은 **20개국 10,000명** 직장인의 직무 환경,
            정신건강 상태, 번아웃 위험도를 수집한 글로벌 설문 데이터입니다 (2020~2024년).
            """
        )

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("총 응답자 수", f"{len(df_dash):,}명")
        kpi2.metric("수집 기간", "2020 ~ 2024년")
        kpi3.metric("대상 국가", f"{df_dash['country'].nunique()}개국")
        kpi4.metric("직무 종류", f"{df_dash['job_role'].nunique()}종")

        st.markdown("---")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 📌 번아웃 위험 점수 분포 (히스토그램)")
            st.caption("번아웃 점수가 어떻게 분포하는지 전체적으로 확인합니다.")
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(
                df_dash["burnout_risk_score"].dropna(),
                bins=30, color="#E07B54", edgecolor="white", alpha=0.85
            )
            mean_val = df_dash["burnout_risk_score"].mean()
            ax.axvline(mean_val, color="#333", linestyle="--",
                       linewidth=1.5, label=f"평균: {mean_val:.2f}")
            ax.set_xlabel("번아웃 위험 점수")
            ax.set_ylabel("인원 수")
            ax.set_title("번아웃 위험 점수 분포")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
            st.info(
                f"평균 번아웃 점수: **{mean_val:.2f}점** | "
                f"최솟값: {df_dash['burnout_risk_score'].min():.1f} / "
                f"최댓값: {df_dash['burnout_risk_score'].max():.1f}\n\n"
                "점수가 1~10 범위에 걸쳐 비교적 고르게 분포하며, 중간 위험군(3~6점)에 집중되어 있습니다."
            )

        with col_b:
            st.markdown("#### 📌 산업군별 응답자 수")
            st.caption("어느 산업에서 가장 많은 응답이 수집됐는지 확인합니다.")
            industry_cnt = df_dash["industry"].value_counts()
            fig, ax = plt.subplots(figsize=(7, 4))
            colors_ind = sns.color_palette("husl", len(industry_cnt))
            ax.barh(industry_cnt.index[::-1], industry_cnt.values[::-1], color=colors_ind)
            ax.set_xlabel("응답자 수")
            ax.set_title("산업군별 응답자 수")
            st.pyplot(fig)
            plt.close(fig)
            st.info(
                f"가장 많은 응답자 산업: **{industry_cnt.index[0]}** ({industry_cnt.iloc[0]:,}명)\n\n"
                "전체 14개 산업군이 비교적 균등하게 분포되어 편향 없이 분석이 가능합니다."
            )

        st.markdown("---")
        st.markdown("#### 📋 수치형 변수 기초 통계량")
        num_desc = df_dash[numeric_cols].describe().T.round(2)
        st.dataframe(num_desc, use_container_width=True)
        st.caption(
            "count: 유효 데이터 수 / mean: 평균 / std: 표준편차 / "
            "min~max: 최솟값·최댓값 / 25%·50%·75%: 사분위수"
        )

    # ============================================================
    # Sub Tab 2. 번아웃 그룹 분포
    # ============================================================
    with sub_tab2:
        st.subheader("🔥 번아웃 위험 그룹 분포 분석")
        st.markdown(
            """
            > 번아웃 위험 점수(1~10점)를 세 구간으로 분류합니다:
            > - **Low (저위험)**: 1~3점 / **Medium (중위험)**: 3~6점 / **High (고위험)**: 6~10점
            """
        )

        group_count = df_dash["burnout_group"].value_counts().reindex(["Low", "Medium", "High"])
        group_pct = (group_count / group_count.sum() * 100).round(1)

        m1, m2, m3 = st.columns(3)
        m1.metric("🟢 저위험 (Low)", f"{group_count.get('Low', 0):,}명",
                  f"{group_pct.get('Low', 0):.1f}%")
        m2.metric("🟡 중위험 (Medium)", f"{group_count.get('Medium', 0):,}명",
                  f"{group_pct.get('Medium', 0):.1f}%")
        m3.metric("🔴 고위험 (High)", f"{group_count.get('High', 0):,}명",
                  f"{group_pct.get('High', 0):.1f}%")

        st.markdown("---")
        col_a, col_b = st.columns(2)
        colors_grp = ["#4CAF50", "#FFC107", "#F44336"]

        with col_a:
            st.markdown("**그룹별 인원 수 (막대 그래프)**")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(["Low", "Medium", "High"], group_count.values,
                   color=colors_grp, edgecolor="white", linewidth=1.2)
            for i, (cnt, pct) in enumerate(zip(group_count.values, group_pct.values)):
                ax.text(i, cnt + 30, f"{cnt:,}\n({pct}%)", ha="center", fontsize=10)
            ax.set_xlabel("번아웃 위험 그룹")
            ax.set_ylabel("인원 수")
            ax.set_title("번아웃 그룹별 인원 분포")
            ax.set_ylim(0, group_count.max() * 1.18)
            st.pyplot(fig)
            plt.close(fig)

        with col_b:
            st.markdown("**그룹별 비율 (파이 차트)**")
            fig, ax = plt.subplots(figsize=(6, 4))
            wedges, texts, autotexts = ax.pie(
                group_count.values,
                labels=["Low", "Medium", "High"],
                colors=colors_grp,
                autopct="%1.1f%%",
                startangle=90,
                pctdistance=0.75
            )
            for at in autotexts:
                at.set_fontsize(11)
                at.set_fontweight("bold")
            ax.set_title("번아웃 그룹 비율")
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")
        st.markdown("#### 📌 그룹별 주요 수치 비교 (박스플롯)")
        st.caption("번아웃 그룹에 따라 주요 지표가 어떻게 달라지는지 확인합니다.")

        stress_map = {"Very Low": 1, "Low": 2, "Moderate": 3, "High": 4, "Very Severe": 5}
        df_box = df_dash.copy()
        df_box["stress_level_num"] = df_box["stress_level"].map(stress_map)
        box_data_cols = ["weekly_overtime_hours", "work_life_balance_score",
                         "job_satisfaction_score", "stress_level_num"]
        box_labels = ["주간 초과근무(h)", "워라밸 점수", "직무 만족도", "스트레스 수준(수치화)"]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, col, label in zip(axes, box_data_cols, box_labels):
            data_by_group = [
                df_box.loc[df_box["burnout_group"] == g, col].dropna()
                for g in ["Low", "Medium", "High"]
            ]
            bp = ax.boxplot(data_by_group, tick_labels=["Low", "Med", "High"], patch_artist=True)
            for patch, color in zip(bp["boxes"], colors_grp):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_title(label, fontsize=10)
            ax.set_xlabel("번아웃 그룹")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            "**핵심 인사이트**: 고위험(High) 그룹은 저위험(Low) 그룹 대비 "
            "**초과근무 시간이 많고**, **워라밸·직무만족도가 낮으며**, "
            "**스트레스 수준이 높은** 경향이 뚜렷하게 나타납니다."
        )

    # ============================================================
    # Sub Tab 3. 변수 상관관계
    # ============================================================
    with sub_tab3:
        st.subheader("🔗 수치형 변수 간 상관관계 (히트맵)")
        st.markdown(
            """
            > 피어슨 상관계수(-1 ~ +1)로 변수 간 선형 관계를 시각화합니다.
            > +1 → 함께 증가 / -1 → 반대로 변화 / 0 → 상관 없음
            """
        )

        corr_cols = [c for c in numeric_cols if c in df_dash.columns]
        corr_matrix = df_dash[corr_cols + ["burnout_risk_score"]].corr()

        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f",
            cmap="coolwarm", square=True, ax=ax,
            linewidths=0.5, annot_kws={"size": 8}
        )
        ax.set_title("수치형 피처 간 상관관계 히트맵", fontsize=13)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        st.markdown("#### 📌 번아웃 점수와 각 변수의 상관계수 순위")
        corr_with_target = (
            corr_matrix["burnout_risk_score"]
            .drop("burnout_risk_score")
            .sort_values(key=abs, ascending=False)
        )
        corr_df = pd.DataFrame({
            "변수": corr_with_target.index,
            "상관계수": corr_with_target.values.round(3),
            "방향": ["🔴 양의 상관" if v > 0 else "🔵 음의 상관" for v in corr_with_target.values]
        })
        st.dataframe(corr_df, use_container_width=True)
        st.info(
            "**주요 발견**: 초과근무 시간(weekly_overtime_hours)이 번아웃 점수와 "
            "가장 강한 양의 상관관계를 보입니다. 반면 워라밸 점수, 직무 만족도는 "
            "음의 상관관계로, 이 값들이 높을수록 번아웃 위험이 낮아집니다."
        )

    # ============================================================
    # Sub Tab 4. 모델 성능 비교
    # ============================================================
    with sub_tab4:
        st.subheader("🏆 머신러닝 모델 성능 비교")
        st.markdown(
            """
            > 4개 회귀 모델을 학습하여 번아웃 점수를 예측하고 성능을 비교합니다.
            > - **R²**: 1에 가까울수록 좋음 / **RMSE·MAE**: 낮을수록 정확
            """
        )

        try:
            from sklearn.model_selection import train_test_split
            from sklearn.linear_model import LinearRegression, ElasticNet
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

            X_all = df_dash[numeric_cols + list(categorical_cols)].copy()
            y_all = df_dash["burnout_risk_score"]

            X_cat_all = encoder.transform(X_all[list(categorical_cols)])
            X_cat_df_all = pd.DataFrame(
                X_cat_all, columns=encoder.get_feature_names_out(list(categorical_cols))
            )
            X_num_all = X_all[numeric_cols].reset_index(drop=True)
            X_enc_all = pd.concat([X_num_all, X_cat_df_all], axis=1).reindex(
                columns=feature_cols, fill_value=0
            )
            X_scaled_all = scaler.transform(X_enc_all)

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_scaled_all, y_all, test_size=0.2, random_state=42
            )

            eval_models = {
                "Linear Regression": LinearRegression(),
                "Elastic Net": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
                "XGBoost": model,
            }

            perf_rows = []
            y_pred_xgb_plot = None
            y_te_plot = np.array(y_te)

            for name, m in eval_models.items():
                if name == "XGBoost":
                    y_pred = m.predict(X_te)
                    y_pred_xgb_plot = y_pred
                else:
                    m.fit(X_tr, y_tr)
                    y_pred = m.predict(X_te)
                r2 = r2_score(y_te, y_pred)
                rmse = np.sqrt(mean_squared_error(y_te, y_pred))
                mae = mean_absolute_error(y_te, y_pred)
                perf_rows.append({"모델": name, "R²": round(r2, 4),
                                  "RMSE": round(rmse, 4), "MAE": round(mae, 4)})

            perf_df = pd.DataFrame(perf_rows).sort_values("R²", ascending=False)

            st.markdown("#### 📋 모델별 성능 지표")
            st.dataframe(
                perf_df.style
                    .highlight_max(subset=["R²"], color="#c6f1d6")
                    .highlight_min(subset=["RMSE", "MAE"], color="#c6f1d6"),
                use_container_width=True
            )

            st.markdown("---")
            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown("#### R² Score 비교")
                fig, ax = plt.subplots(figsize=(6, 4))
                bar_colors_r2 = [
                    "#4CAF50" if v == perf_df["R²"].max() else "#90CAF9"
                    for v in perf_df["R²"]
                ]
                bars = ax.bar(perf_df["모델"], perf_df["R²"],
                              color=bar_colors_r2, edgecolor="white")
                for bar, val in zip(bars, perf_df["R²"]):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.005,
                            f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
                ax.set_xlabel("모델")
                ax.set_ylabel("R² Score")
                ax.set_title("모델별 R² Score 비교")
                ax.set_ylim(0, 1.05)
                ax.tick_params(axis="x", rotation=15)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with col_b:
                st.markdown("#### RMSE 비교 (낮을수록 좋음)")
                fig, ax = plt.subplots(figsize=(6, 4))
                bar_colors_rmse = [
                    "#F44336" if v == perf_df["RMSE"].min() else "#FFCCBC"
                    for v in perf_df["RMSE"]
                ]
                bars2 = ax.bar(perf_df["모델"], perf_df["RMSE"],
                               color=bar_colors_rmse, edgecolor="white")
                for bar, val in zip(bars2, perf_df["RMSE"]):
                    ax.text(bar.get_x() + bar.get_width()/2,
                            bar.get_height() + 0.005,
                            f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
                ax.set_xlabel("모델")
                ax.set_ylabel("RMSE")
                ax.set_title("모델별 RMSE 비교")
                ax.tick_params(axis="x", rotation=15)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("---")
            st.markdown("#### 🎯 XGBoost: 실제값 vs 예측값 산점도")
            st.caption("점들이 대각선(y=x)에 가까울수록 예측이 정확함을 의미합니다.")

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(y_te_plot, y_pred_xgb_plot, alpha=0.4,
                       color="#2ca02c", edgecolors="white", s=20, label="예측 결과")
            mn = min(y_te_plot.min(), y_pred_xgb_plot.min())
            mx = max(y_te_plot.max(), y_pred_xgb_plot.max())
            ax.plot([mn, mx], [mn, mx], "r--", linewidth=2, label="완벽한 예측 (y=x)")
            ax.set_xlabel("실제 번아웃 점수", fontsize=12)
            ax.set_ylabel("예측 번아웃 점수", fontsize=12)
            ax.set_title("XGBoost: 실제 vs 예측 비교", fontsize=13, fontweight="bold")
            ax.legend()
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            best_r2 = perf_df["R²"].max()
            best_name = perf_df.loc[perf_df["R²"].idxmax(), "모델"]
            st.success(
                f"**최고 성능 모델: {best_name}** (R² = {best_r2:.4f})\n\n"
                f"모델이 번아웃 점수 변동의 약 **{best_r2*100:.1f}%** 를 설명합니다."
            )

        except Exception as e:
            st.error(f"모델 성능 재계산 중 오류: {e}")

    # ============================================================
    # Sub Tab 5. 피처 중요도
    # ============================================================
    with sub_tab5:
        st.subheader("🎯 XGBoost 피처 중요도 (Feature Importance)")
        st.markdown(
            """
            > 모델이 번아웃 점수를 예측할 때 각 변수에 얼마나 의존하는지를 나타냅니다.
            > 중요도가 높을수록 해당 변수가 예측에 결정적인 역할을 합니다.
            """
        )

        try:
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                "Feature": feature_cols,
                "Importance": importances
            }).sort_values("Importance", ascending=False).reset_index(drop=True)

            top15 = importance_df.head(15)
            palette_imp = sns.color_palette("YlOrRd_r", len(top15))

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(data=top15, x="Importance", y="Feature",
                        palette=palette_imp, ax=ax)
            ax.set_title("XGBoost Top 15 Feature Importances", fontsize=13, fontweight="bold")
            ax.set_xlabel("중요도 (Importance Score)")
            ax.set_ylabel("")
            for i, (_, row) in enumerate(top15.iterrows()):
                ax.text(row["Importance"] + 0.0005, i,
                        f"{row['Importance']:.4f}", va="center", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("---")
            col_a, col_b = st.columns([1, 1.5])

            with col_a:
                st.markdown("#### 📋 Top 20 피처 중요도 순위")
                top20 = importance_df.head(20).copy()
                top20.index = range(1, len(top20) + 1)
                top20.index.name = "순위"
                st.dataframe(top20, use_container_width=True)

            with col_b:
                top3 = top15.head(3)["Feature"].tolist()
                st.markdown("#### 💡 핵심 해석")
                st.markdown(
                    f"**상위 3개 피처 분석:**\n\n"
                    f"1. **{top3[0]}** — 번아웃 예측의 가장 핵심 변수입니다. "
                    f"초과근무 시간이 길수록 번아웃 위험이 급격히 증가합니다.\n\n"
                    f"2. **{top3[1]}** — 워라밸(일·생활 균형) 점수로, "
                    f"점수가 낮을수록 번아웃 위험이 높아지는 강한 음의 영향력을 가집니다.\n\n"
                    f"3. **{top3[2]}** — 연간 결근일 수는 "
                    f"번아웃의 결과이자 원인으로, 번아웃이 심할수록 결근이 늘어납니다.\n\n"
                    "**종합 시사점:** 번아웃 예방을 위해서는 **초과근무 축소**, "
                    "**워라밸 개선**, **스트레스 관리**가 가장 효과적입니다."
                )

        except Exception as e:
            st.error(f"피처 중요도 표시 오류: {e}")

    # ============================================================
    # Sub Tab 6. 정신건강 진단 분석
    # ============================================================
    with sub_tab6:
        st.subheader("🏥 번아웃 그룹별 정신건강 진단 분석")
        st.markdown(
            """
            > 번아웃 위험 그룹(Low/Medium/High)에 따라 **정신건강 진단 경험**과
            **진단 유형**이 어떻게 달라지는지 분석합니다.
            """
        )

        diagnosis_by_burnout = pd.crosstab(
            df_dash["burnout_group"],
            df_dash["has_diagnosis"],
            normalize="index"
        ) * 100

        st.markdown("#### 📌 번아웃 그룹별 정신건강 진단 경험 비율")

        yes_rate = diagnosis_by_burnout.get("Yes", pd.Series(dtype=float))

        if len(yes_rate) > 0:
            low_r = yes_rate.get("Low", 0)
            med_r = yes_rate.get("Medium", 0)
            high_r = yes_rate.get("High", 0)
            m1, m2, m3 = st.columns(3)
            m1.metric("🟢 Low 그룹 진단율", f"{low_r:.1f}%")
            m2.metric("🟡 Medium 그룹 진단율", f"{med_r:.1f}%",
                      delta=f"+{med_r - low_r:.1f}%p vs Low")
            m3.metric("🔴 High 그룹 진단율", f"{high_r:.1f}%",
                      delta=f"+{high_r - low_r:.1f}%p vs Low")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**그룹별 진단 비율 (막대)**")
            fig, ax = plt.subplots(figsize=(6, 4))
            diagnosis_by_burnout.plot(kind="bar", ax=ax,
                                      color=["#EF9A9A", "#66BB6A"], edgecolor="white")
            ax.set_title("번아웃 그룹별 정신건강 진단 비율")
            ax.set_ylabel("비율 (%)")
            ax.set_xlabel("번아웃 그룹")
            ax.tick_params(axis="x", rotation=0)
            ax.legend(title="진단 여부", labels=["진단 없음 (No)", "진단 있음 (Yes)"])
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_b:
            diag_display = diagnosis_by_burnout.round(1).copy()
            diag_display.columns = ["진단 없음 (%)", "진단 있음 (%)"]
            diag_display.index.name = "번아웃 그룹"
            st.markdown("**정신건강 진단 비율 상세**")
            st.dataframe(diag_display, use_container_width=True)
            if len(yes_rate) > 0:
                st.info(
                    f"번아웃 위험이 높아질수록 정신건강 진단 경험 비율이 "
                    f"**{low_r:.1f}% → {med_r:.1f}% → {high_r:.1f}%** 로 단계적으로 증가합니다."
                )

        st.markdown("---")
        st.markdown("#### 📌 번아웃 그룹별 정신건강 문제 유형")
        st.caption("각 번아웃 그룹에서 어떤 정신건강 문제 유형이 주로 나타나는지 분석합니다.")

        condition_by_burnout = pd.crosstab(
            df_dash["burnout_group"],
            df_dash["mental_health_condition"],
            normalize="index"
        ) * 100

        col_c, col_d = st.columns(2)

        with col_c:
            fig, ax = plt.subplots(figsize=(7, 5))
            condition_by_burnout.plot(kind="bar", stacked=True, ax=ax,
                                      colormap="tab10", edgecolor="white", linewidth=0.5)
            ax.set_title("번아웃 그룹별 정신건강 문제 유형")
            ax.set_ylabel("비율 (%)")
            ax.set_xlabel("번아웃 그룹")
            ax.tick_params(axis="x", rotation=0)
            ax.legend(title="정신건강 유형", bbox_to_anchor=(1.02, 1),
                      loc="upper left", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_d:
            st.dataframe(condition_by_burnout.round(1), use_container_width=True)

        st.info(
            "**핵심 해석**\n\n"
            "- **Low 그룹**: 'Healthy(건강)' 비율이 높고, Anxiety·Depression 등 경증이 주를 이룹니다.\n"
            "- **Medium 그룹**: Anxiety, Depression, Burnout이 혼재하며 복합 진단도 증가합니다.\n"
            "- **High 그룹**: Burnout 및 Multiple Conditions(복합 진단) 비율이 크게 증가하며, "
            "전문 치료가 필요한 상태가 많아집니다."
        )

    # ============================================================
    # Sub Tab 7. 주요 변수 심층 분석
    # ============================================================
    with sub_tab7:
        st.subheader("🔍 주요 변수별 심층 분석")
        st.markdown(
            """
            > 번아웃에 큰 영향을 미치는 핵심 변수들을 심층적으로 분석합니다.
            """
        )

        # 초과근무 vs 번아웃 산점도
        st.markdown("#### 📌 주간 초과근무 시간 vs 번아웃 점수")
        col_a, col_b = st.columns(2)

        with col_a:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(
                df_dash["weekly_overtime_hours"],
                df_dash["burnout_risk_score"],
                alpha=0.15, s=8, color="#E07B54"
            )
            ot_clean = df_dash["weekly_overtime_hours"].dropna()
            bs_clean = df_dash.loc[ot_clean.index, "burnout_risk_score"]
            z = np.polyfit(ot_clean, bs_clean, 1)
            p = np.poly1d(z)
            xp = np.linspace(ot_clean.min(), ot_clean.max(), 100)
            ax.plot(xp, p(xp), "r-", linewidth=2, label="추세선")
            ax.set_xlabel("주간 초과근무 시간 (h)")
            ax.set_ylabel("번아웃 위험 점수")
            ax.set_title("초과근무 시간 vs 번아웃 점수")
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)

        with col_b:
            ot_corr = df_dash["weekly_overtime_hours"].corr(df_dash["burnout_risk_score"])
            st.markdown(
                f"**분석 결과:**\n\n"
                f"- 상관계수: **r = {ot_corr:.3f}**\n"
                f"- 초과근무 1시간 증가 시 번아웃 점수 평균 약 **{ot_corr * 2:.2f}점** 상승\n"
                f"- 주 20시간 이상 초과근무 그룹은 고위험(High) 비율이 급증\n\n"
                "⚠️ **초과근무 시간은 번아웃의 가장 강력한 예측 변수**입니다."
            )

        st.markdown("---")

        # 스트레스 수준별 번아웃
        st.markdown("#### 📌 스트레스 수준별 번아웃 점수 분포")
        col_c, col_d = st.columns(2)
        stress_order = ["Very Low", "Low", "Moderate", "High", "Very Severe"]
        stress_exist = [s for s in stress_order if s in df_dash["stress_level"].values]

        with col_c:
            fig, ax = plt.subplots(figsize=(7, 4))
            data_stress = [
                df_dash.loc[df_dash["stress_level"] == s, "burnout_risk_score"].dropna()
                for s in stress_exist
            ]
            bp = ax.boxplot(data_stress, labels=stress_exist, patch_artist=True)
            s_colors = ["#81C784", "#AED581", "#FFF176", "#FFB74D", "#EF5350"]
            for patch, color in zip(bp["boxes"], s_colors[:len(stress_exist)]):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            ax.set_xlabel("스트레스 수준")
            ax.set_ylabel("번아웃 위험 점수")
            ax.set_title("스트레스 수준별 번아웃 점수 분포")
            ax.tick_params(axis="x", rotation=15)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_d:
            stress_means = df_dash.groupby("stress_level")["burnout_risk_score"].mean()
            stress_means = stress_means.reindex(stress_exist)
            st.markdown("**스트레스 수준별 평균 번아웃 점수**")
            s_disp = pd.DataFrame({
                "스트레스 수준": stress_means.index,
                "평균 번아웃 점수": stress_means.values.round(2)
            })
            st.dataframe(s_disp, use_container_width=True)
            st.info(
                "스트레스 수준이 올라갈수록 번아웃 점수가 단계적으로 상승합니다. "
                "스트레스 관리는 번아웃 예방의 핵심 요소입니다."
            )

        st.markdown("---")

        # 근무 방식·고용 형태별 비교
        st.markdown("#### 📌 근무 방식 및 고용 형태별 번아웃 비교")
        col_e, col_f = st.columns(2)

        with col_e:
            wm_means = df_dash.groupby("work_model")["burnout_risk_score"].mean().sort_values()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(wm_means.index, wm_means.values,
                    color=sns.color_palette("coolwarm", len(wm_means)))
            for i, v in enumerate(wm_means.values):
                ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=10)
            ax.set_xlabel("평균 번아웃 점수")
            ax.set_title("근무 방식별 평균 번아웃 점수")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_f:
            et_means = df_dash.groupby("employment_type")["burnout_risk_score"].mean().sort_values()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(et_means.index, et_means.values,
                    color=sns.color_palette("RdYlGn_r", len(et_means)))
            for i, v in enumerate(et_means.values):
                ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=10)
            ax.set_xlabel("평균 번아웃 점수")
            ax.set_title("고용 형태별 평균 번아웃 점수")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")

        # 국가별 번아웃
        st.markdown("#### 📌 국가별 평균 번아웃 점수")
        country_means = (
            df_dash.groupby("country")["burnout_risk_score"].mean()
            .sort_values(ascending=False)
        )
        global_mean = country_means.mean()

        fig, ax = plt.subplots(figsize=(12, 4))
        bar_c = ["#E53935" if v > global_mean else "#42A5F5" for v in country_means.values]
        ax.bar(country_means.index, country_means.values, color=bar_c, edgecolor="white")
        ax.axhline(global_mean, color="black", linestyle="--", linewidth=1.5,
                   label=f"전체 평균: {global_mean:.2f}")
        ax.set_xlabel("국가")
        ax.set_ylabel("평균 번아웃 점수")
        ax.set_title("국가별 평균 번아웃 점수 (빨간색: 평균 이상)")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.info(
            f"**가장 높은 번아웃**: {country_means.index[0]} ({country_means.iloc[0]:.2f}점) | "
            f"**가장 낮은 번아웃**: {country_means.index[-1]} ({country_means.iloc[-1]:.2f}점)\n\n"
            "국가별 차이는 근무 문화, 사회 안전망, 정신건강 정책의 차이를 반영합니다."
        )

