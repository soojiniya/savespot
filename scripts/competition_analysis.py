"""공모전 설명용 분석 산출물 생성 (읽기 전용 분석 스크립트).

기존 모델·예측 결과·서비스 데이터는 읽기만 하고 수정하지 않는다.
산출물은 모두 results/competition/ 아래에 새로 만든다.

    python scripts/competition_analysis.py

1. 모델 설명 가능성: 최종 Logistic Regression 계수(StandardScaler 이후 표준화 계수)
2. 대표 지역 사례 3개: 시설 데이터가 있는 49개 지역 안에서 규칙으로 선정
3. 개선 검토 점수 가중치 민감도: 현재 산식(노트북 14)을 기준안으로 비교
4. 종합 요약 보고서
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy.stats import kendalltau, spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "competition"
OUT.mkdir(parents=True, exist_ok=True)

PRED_PATH = ROOT / "results" / "predicted_high_risk_2026_229.csv"
BLIND_PATH = ROOT / "results" / "infrastructure_blind_spots_49.csv"
MODEL_PATH = ROOT / "models" / "high_risk_classifier.joblib"
EDA_PATH = ROOT / "data_processed" / "eda_main_49.csv"
BUMP_PATH = ROOT / "data_processed" / "final_speed_bump_35.csv"
PANEL_PATH = ROOT / "data_processed" / "panel_2020_2025_229.csv"
PERM_PATH = ROOT / "results" / "feature_importance_permutation_2025.csv"
METRICS_PATH = ROOT / "results" / "model_average_metrics.csv"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


INPUT_HASHES_BEFORE = {p.name: sha256(p) for p in (PRED_PATH, BLIND_PATH, MODEL_PATH)}

# ---------------------------------------------------------------- style
# 한글 글꼴: Windows(맑은 고딕) → macOS → 나눔 → Noto CJK(한글 포함)
for family in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans CJK JP"):
    if any(f.name == family for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = family
        break
plt.rcParams["axes.unicode_minus"] = False
INK, MUTED, GRID = "#1f2328", "#6b7076", "#e6e8eb"
POS, NEG, NEUTRAL = "#d9534f", "#2f7fbf", "#9aa0a6"  # 고위험 방향 / 저위험 방향
TEAL = "#1a9e86"
CASE_COLORS = ["#d9534f", "#e3a21a", "#2f7fbf"]
plt.rcParams.update({
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK,
    "axes.titleweight": "bold", "axes.titlesize": 13, "font.size": 10.5,
})


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def score100(p: float) -> str:
    return f"{p * 100:.1f}점"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(lines)


FEATURE_KO = {
    "prev_child_population": "전년 어린이 인구(0~12세)",
    "prev_vehicle_count": "전년 자동차 등록대수",
    "prev_accident_count": "전년 어린이 교통사고 건수",
    "prev_injury_count": "전년 어린이 부상자 수",
    "prev_death_count": "전년 어린이 사망자 수",
    "prev_accident_per_10k_child": "전년 어린이 1만 명당 사고율",
    "prev_vehicles_per_100_child": "전년 어린이 100명당 자동차 수",
    "accident_count_change_1y": "사고 건수 1년 증감",
    "accident_rate_change_1y": "사고율 1년 증감",
    "accident_count_mean_2y": "최근 2년 평균 사고 건수",
}
FEATURE_NOTE = {
    "prev_child_population": "다른 입력이 같을 때 어린이 인구가 많을수록 저위험 방향. 고위험 기준이 인구 대비 사고율(상위 25%)이라 같은 사고 건수라도 인구가 크면 비율이 낮아지는 구조가 반영된 것으로 보임.",
    "prev_vehicle_count": "계수가 0에 가까워 판단에 주는 영향이 매우 작음.",
    "prev_accident_count": "전년 사고 건수가 많을수록 고위험 방향.",
    "prev_injury_count": "전년 부상자 수가 많을수록 고위험 방향. 사고 건수와 함께 움직이는 변수라 단독 해석은 주의.",
    "prev_death_count": "계수가 작고 사망 건수 자체가 드물어 판단에 주는 영향이 작음.",
    "prev_accident_per_10k_child": "전년 인구 대비 사고율이 높을수록 고위험 방향(목표 정의와 같은 지표의 전년 값).",
    "prev_vehicles_per_100_child": "계수가 0에 가까워 판단에 주는 영향이 매우 작음.",
    "accident_count_change_1y": "사고 수준이 같다면 직전 1년 증가폭이 클수록 약하게 저위험 방향. 급증한 해 다음에는 평균 쪽으로 돌아오는 경향이 반영된 것으로 보이며, 증가가 '좋다'는 뜻이 아님.",
    "accident_rate_change_1y": "사고율 수준이 같다면 직전 1년 사고율 증가폭이 클수록 저위험 방향. 위와 같은 평균 회귀 성격으로 해석하며 단독 해석은 금물.",
    "accident_count_mean_2y": "최근 2년 평균 사고 건수가 많을수록 고위험 방향. 일시적 변동보다 지속적 사고 수준을 반영.",
}

# =====================================================================
# 1. 모델 설명 가능성
# =====================================================================
bundle = joblib.load(MODEL_PATH)
pipe, features, model_name = bundle["model"], list(bundle["feature_cols"]), bundle["model_name"]
step_names = [name for name, _ in pipe.steps]
scaler = dict(pipe.steps).get("scaler")
lr = pipe.steps[-1][1]
coef = lr.coef_[0]
intercept = float(lr.intercept_[0])

pred = read(PRED_PATH)
# 저장된 2026 예측값이 현재 모델로 그대로 재현되는지 확인 (값을 바꾸지 않음)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    reproduced = pipe.predict_proba(pred[features])[:, 1]
max_repro_diff = float(np.max(np.abs(reproduced - pred["predicted_high_risk_probability_2026"])))

# 학습 데이터 재구성(노트북 13과 같은 규칙) → 변수 간 상관 확인용
panel = read(PANEL_PATH).sort_values(["sido_sgg_nm", "year"]).copy()
g = panel.groupby("sido_sgg_nm")
for src, dst in [("child_population_0_12", "prev_child_population"), ("vehicle_count", "prev_vehicle_count"),
                 ("accident_count", "prev_accident_count"), ("injury_count", "prev_injury_count"),
                 ("death_count", "prev_death_count"), ("accident_per_10k_child", "prev_accident_per_10k_child"),
                 ("vehicles_per_100_child", "prev_vehicles_per_100_child")]:
    panel[dst] = g[src].shift(1)
panel["prev2_accident_count"] = g["accident_count"].shift(2)
panel["prev2_accident_rate"] = g["accident_per_10k_child"].shift(2)
panel["accident_count_change_1y"] = panel["prev_accident_count"] - panel["prev2_accident_count"]
panel["accident_rate_change_1y"] = panel["prev_accident_per_10k_child"] - panel["prev2_accident_rate"]
panel["accident_count_mean_2y"] = (panel["prev_accident_count"] + panel["prev2_accident_count"]) / 2
train = panel[panel["year"].between(2022, 2025)].dropna(subset=features)
corr = train[features].corr()
train_rows = len(train)

perm = read(PERM_PATH).set_index("feature")
expl = pd.DataFrame({
    "feature": features,
    "feature_ko": [FEATURE_KO.get(f, f) for f in features],
    "coefficient_standardized": coef,
})
expl["abs_coefficient"] = expl["coefficient_standardized"].abs()
expl["abs_rank"] = expl["abs_coefficient"].rank(ascending=False, method="min").astype(int)
expl["direction"] = np.where(expl["coefficient_standardized"] > 0, "고위험 방향(+)", "저위험 방향(-)")
expl["odds_ratio_per_1sd"] = np.exp(expl["coefficient_standardized"])
if scaler is not None:
    expl["scaler_mean"] = scaler.mean_
    expl["scaler_std"] = scaler.scale_
    expl["coefficient_per_original_unit"] = expl["coefficient_standardized"] / scaler.scale_
expl["permutation_importance_2025"] = expl["feature"].map(perm["importance_mean"])
expl["max_abs_corr_with_other_feature"] = [
    float(corr.loc[f].drop(f).abs().max()) for f in features
]
expl["most_correlated_feature"] = [FEATURE_KO[corr.loc[f].drop(f).abs().idxmax()] for f in features]
expl["interpretation"] = expl["feature"].map(FEATURE_NOTE)
expl = expl.sort_values("abs_rank").reset_index(drop=True)
expl.to_csv(OUT / "model_explainability.csv", index=False, encoding="utf-8-sig")

# 그래프 1: 표준화 계수 막대
fig, ax = plt.subplots(figsize=(9.5, 6))
d = expl.iloc[::-1]
colors = [POS if c > 0 else NEG for c in d["coefficient_standardized"]]
ax.barh(d["feature_ko"], d["coefficient_standardized"], color=colors, height=0.62)
ax.axvline(0, color=INK, lw=0.8)
for y, v in enumerate(d["coefficient_standardized"]):
    ax.text(v + (0.06 if v >= 0 else -0.06), y, f"{v:+.2f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=9.5, color=INK)
ax.set_xlim(min(-5.6, d["coefficient_standardized"].min() - 0.8), max(1.9, d["coefficient_standardized"].max() + 0.7))
ax.set_xlabel("표준화 계수 (StandardScaler 이후, 입력 1표준편차 증가 시 log-odds 변화)")
ax.set_title("최종 Logistic Regression 입력변수 계수")
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.text(0.01, 0.02, "빨강: 고위험군 방향  ·  파랑: 저위험군 방향\n상관/모델 판단 방향이며 인과관계가 아님",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=MUTED)
fig.tight_layout()
fig.savefig(OUT / "model_explainability.png", dpi=180)
plt.close(fig)

# 그래프 2: 방향 설명형 (1표준편차 증가 시 odds 배수, 로그축)
fig, ax = plt.subplots(figsize=(10, 6))
d = expl.sort_values("coefficient_standardized")
y = np.arange(len(d))
ors = d["odds_ratio_per_1sd"].to_numpy()
for yi, (o, c) in enumerate(zip(ors, d["coefficient_standardized"])):
    color = POS if c > 0.1 else NEG if c < -0.1 else NEUTRAL
    ax.plot([1, o], [yi, yi], color=color, lw=2.4, solid_capstyle="round")
    ax.scatter([o], [yi], color=color, s=46, zorder=3)
    label = f"×{o:.2f}" if o >= 0.1 else f"×{o:.3f}"
    ax.text(o * (1.12 if o >= 1 else 0.89), yi, label, va="center", ha="left" if o >= 1 else "right",
            fontsize=9.5, color=INK)
ax.axvline(1, color=INK, lw=0.8)
ax.set_xscale("log")
ax.set_yticks(y)
ax.set_yticklabels(d["feature_ko"])
ax.set_xlim(0.004, 12)
ax.set_xticks([0.01, 0.1, 0.5, 1, 2, 5])
ax.set_xticklabels(["0.01", "0.1", "0.5", "1", "2", "5"])
ax.set_xlabel("다른 입력이 같을 때 해당 변수가 1표준편차 높아지면 '고위험군' 판단 odds가 몇 배가 되는가 (로그축)")
ax.set_title("모델 판단 방향: 어떤 값이 높을 때 고위험군 쪽으로 기우는가")
ax.text(0.0045, len(d) - 0.3, "← 저위험군 방향", color=NEG, fontsize=10, va="center", fontweight="bold")
ax.text(11, len(d) - 0.3, "고위험군 방향 →", color=POS, fontsize=10, va="center", ha="right", fontweight="bold")
ax.set_ylim(-0.7, len(d) + 0.2)
ax.grid(axis="x", color=GRID, lw=0.8, which="major")
ax.set_axisbelow(True)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.text(0.99, -0.16, "회색: 영향이 매우 작은 변수(|계수| ≤ 0.1). 사고 관련 변수들은 서로 상관이 높아 개별 계수를 단독으로 해석하지 않는다. 인과관계가 아님.",
        transform=ax.transAxes, ha="right", fontsize=8.5, color=MUTED)
fig.tight_layout()
fig.savefig(OUT / "model_explainability_direction.png", dpi=180)
plt.close(fig)

metrics = read(METRICS_PATH).set_index("model")
lr_m = metrics.loc["Logistic Regression"]
pos = expl[expl["coefficient_standardized"] > 0]
neg = expl[expl["coefficient_standardized"] < 0]
table1 = expl[["abs_rank", "feature_ko", "feature", "coefficient_standardized", "direction", "odds_ratio_per_1sd",
               "permutation_importance_2025"]].copy()
table1["coefficient_standardized"] = table1["coefficient_standardized"].map(lambda v: f"{v:+.3f}")
table1["odds_ratio_per_1sd"] = table1["odds_ratio_per_1sd"].map(lambda v: f"{v:.3f}")
table1["permutation_importance_2025"] = table1["permutation_importance_2025"].map(lambda v: f"{v:.3f}")
table1.columns = ["절댓값 순위", "변수", "컬럼", "표준화 계수", "방향", "1SD당 odds 배수", "순열중요도(2025)"]
high_corr_pairs = []
for i, a in enumerate(features):
    for b in features[i + 1:]:
        if abs(corr.loc[a, b]) >= 0.8:
            high_corr_pairs.append(f"{FEATURE_KO[a]} ↔ {FEATURE_KO[b]} (r = {corr.loc[a, b]:.2f})")

model_md = f"""# 모델 설명 가능성 분석 (Logistic Regression 계수)

> 상관/모델 판단 방향이지 인과관계가 아닙니다. 계수는 "이 값이 높은 지역을 모델이 어느 쪽으로 분류하는 경향이 있는가"를 보여줄 뿐, 그 변수가 사고를 일으키거나 줄인다는 뜻이 아닙니다.

## 1. 최종 모델
- 모델: **{model_name}** (`models/high_risk_classifier.joblib`)
- 구조: `{' → '.join(step_names)}` = StandardScaler → LogisticRegression(class_weight="balanced", max_iter=3000, random_state=42)
- 학습 데이터: 2022~2025년 229개 시군구 패널 {train_rows}행 (노트북 13)
- 목표(라벨): 해당 연도 어린이 1만 명당 사고율이 그 해 229개 지역 중 **상위 25% 이상**이면 1(상대적 고위험군)
- 연도별 순차 검증(walk-forward) 평균: ROC-AUC {lr_m['roc_auc']:.3f}, PR-AUC {lr_m['average_precision']:.3f}, Recall {lr_m['recall']:.3f} — 비교 모델(Random Forest, HistGradientBoosting)보다 PR-AUC·ROC-AUC가 높아 최종 선택
- 절편: {intercept:+.3f}
- 확인: 저장된 모델로 `predicted_high_risk_2026_229.csv`의 입력값을 다시 넣으면 저장된 2026 점수와 최대 차이 {max_repro_diff:.2e} (동일)

## 2. 실제 입력 feature 목록 ({len(features)}개)
{chr(10).join(f'- `{f}` — {FEATURE_KO[f]}' for f in features)}

## 3. 계수 해석 방법
- 모델 앞에 StandardScaler가 있으므로 계수는 **표준화 계수**입니다. 즉 "다른 입력이 같을 때, 이 변수가 학습 데이터 기준 1표준편차 높아지면 고위험군 log-odds가 계수만큼 변한다"는 뜻이라, 변수끼리 크기를 비교할 수 있습니다.
- 원래 단위 기준 계수(`coefficient_per_original_unit` = 표준화 계수 ÷ 표준편차)도 CSV에 함께 넣었습니다.
- `1SD당 odds 배수` = exp(계수). 1보다 크면 고위험군 방향, 1보다 작으면 저위험군 방향입니다.
- 출력값(0~1)은 "상대적 고위험군에 속할 가능성에 대한 모델 점수"이며, 실제 사고가 날 확률이 아닙니다.

## 4. 계수와 방향 (절댓값 순위)
{md_table(table1)}

- **고위험군 방향(+) {len(pos)}개**: {', '.join(pos['feature_ko'])}
- **저위험군 방향(-) {len(neg)}개**: {', '.join(neg['feature_ko'])}

### 변수별 해석
{chr(10).join(f'- **{r.feature_ko}** ({r.coefficient_standardized:+.3f}): {r.interpretation}' for r in expl.itertuples())}

## 5. 해석 시 주의점
1. **인과관계가 아닙니다.** 모델이 과거 자료에서 찾은 판단 방향(상관)입니다.
2. **다중공선성:** 사고 관련 변수들이 서로 강하게 함께 움직입니다(학습 데이터 기준 |r| ≥ 0.8인 쌍):
{chr(10).join(f'   - {p}' for p in high_corr_pairs) if high_corr_pairs else '   - 없음'}
   이런 변수들의 개별 계수 부호·크기는 서로 나눠 가진 결과라 단독으로 해석하지 말고 "사고 수준 변수 묶음이 높을수록 고위험 방향"처럼 묶어서 읽는 것이 안전합니다.
3. **어린이 인구 계수(-)의 의미:** 라벨이 '인구 대비 사고율'이라, 사고 건수가 같다면 인구가 큰 지역은 비율이 낮아집니다. 계수가 크게 음수인 것은 이 계산 구조를 반영한 것이며 "어린이가 많으면 안전하다"는 뜻이 아닙니다.
4. **증감 변수(-):** 직전 1년 사고가 급증한 지역이 다음 해 평균 쪽으로 돌아오는 경향(평균 회귀)이 반영된 것으로 보입니다.
5. **순열중요도와 순서가 다를 수 있습니다.** 계수는 '방향과 민감도', 순열중요도(`feature_importance_permutation_2025.csv`)는 '그 변수를 섞었을 때 성능이 얼마나 떨어지는가'를 봅니다. 둘 다 사고 수준·인구 관련 변수가 상위라는 점은 같습니다.
6. 시설(횡단보도·신호등·보호구역·CCTV) 데이터는 **모델 입력이 아닙니다.** 시설 부족도는 49개 지역에서 별도로 계산합니다.

산출물: `model_explainability.csv`, `model_explainability.png`(계수 막대), `model_explainability_direction.png`(방향 설명형)
"""
(OUT / "model_explainability_summary.md").write_text(model_md, encoding="utf-8")

# =====================================================================
# 2. 대표 지역 사례 3개 (49개 시설분석 지역 안에서 규칙 기반 선정)
# =====================================================================
blind = read(BLIND_PATH)
eda = read(EDA_PATH)
bump = read(BUMP_PATH)[["sido_sgg_nm", "speed_bump_count"]]
pred_cols = ["sido_sgg_nm", "risk_rank_2026"] + features
base = (blind.merge(pred[pred_cols], on="sido_sgg_nm", how="left", validate="one_to_one")
        .merge(eda[["sido_sgg_nm", "child_population_0_12", "child_accident_count", "child_injury_count",
                    "child_death_count", "accident_per_10k_child", "crosswalk_count", "traffic_light_count",
                    "schoolzone_count", "cctv_installed_zone_count", "schoolzone_data_available"]],
               on="sido_sgg_nm", how="left", validate="one_to_one")
        .merge(bump, on="sido_sgg_nm", how="left", validate="one_to_one"))
assert len(base) == 49 and base["risk_rank_2026"].notna().all()

ai = base["predicted_high_risk_probability_2026"]
dfc = base["infrastructure_deficiency_score"]
ai_q1, ai_q3 = ai.quantile(0.25), ai.quantile(0.75)
def_med, def_q3 = dfc.quantile(0.5), dfc.quantile(0.75)
base["ai_rank_in_49"] = ai.rank(ascending=False, method="min").astype(int)
base["deficiency_rank_in_49"] = dfc.rank(ascending=False, method="min").astype(int)

rules = {
    "A": ("AI 상대적 고위험도 높음 + 시설 부족도 높음",
          f"49개 중 AI 점수 상위 25%(≥{ai_q3:.3f}) 지역 가운데 시설 부족도가 가장 높은 지역",
          base[ai >= ai_q3].sort_values(["infrastructure_deficiency_score", "improvement_priority_score"], ascending=False)),
    "B": ("AI 상대적 고위험도 높음 + 시설 부족도 상대적으로 낮음",
          f"49개 중 AI 점수 상위 25%(≥{ai_q3:.3f}) 이면서 시설 부족도가 49개 중앙값({def_med:.3f}) 미만인 지역 가운데 AI 점수가 가장 높은 지역",
          base[(ai >= ai_q3) & (dfc < def_med)].sort_values("predicted_high_risk_probability_2026", ascending=False)),
    "C": ("AI 상대적 고위험도 중간 + 시설 부족도 높음",
          f"AI 점수가 49개 중 중간 구간(25~75%: {ai_q1:.3f}~{ai_q3:.3f}) 이고 시설 부족도가 상위 25%(≥{def_q3:.3f})인 지역 가운데 개선 검토 점수가 가장 높은 지역",
          base[(ai > ai_q1) & (ai < ai_q3) & (dfc >= def_q3)].sort_values("improvement_priority_score", ascending=False)),
}
candidates_note = {}
for key, (_, _, frame) in rules.items():
    candidates_note[key] = ", ".join(f"{n.split()[-1]}" for n in frame["sido_sgg_nm"].head(4))

# 정선군·태백시·태안군 확인
check_named = []
for name in ["강원특별자치도 정선군", "강원특별자치도 태백시", "충청남도 태안군"]:
    r = base[base["sido_sgg_nm"] == name].iloc[0]
    fits = [k for k, (_, _, frame) in rules.items() if name in set(frame["sido_sgg_nm"])]
    check_named.append(
        f"- {name}: AI {score100(r.predicted_high_risk_probability_2026)}(49개 중 {r.ai_rank_in_49}위), "
        f"시설 부족도 {r.infrastructure_deficiency_score:.3f}(49개 중 {r.deficiency_rank_in_49}위), "
        f"개선 검토 {r.improvement_priority_rank}위 → 조건 충족 유형: {', '.join(fits) if fits else '없음'}"
    )

picked = {k: frame.iloc[0] for k, (_, _, frame) in rules.items()}
assert len({r["sido_sgg_nm"] for r in picked.values()}) == 3

# 모델 입력 중 특징적인 값: 표준화 값 × 계수 = log-odds 기여
def contributions(row: pd.Series) -> pd.DataFrame:
    x = row[features].astype(float).to_numpy()
    z = (x - scaler.mean_) / scaler.scale_
    out = pd.DataFrame({"feature": features, "value": x, "z": z, "contribution": z * coef})
    out["feature_ko"] = out["feature"].map(FEATURE_KO)
    return out.reindex(out["contribution"].abs().sort_values(ascending=False).index)


def pct49(col: str, value: float) -> float:
    return float((base[col] <= value).mean() * 100)


def fmt_int(v) -> str:
    return "데이터 미제공" if pd.isna(v) else f"{int(round(v)):,}"


rows, case_md = [], []
for key, row in picked.items():
    title, rule_text, _ = rules[key]
    c = contributions(row)
    top_pos = c[c["contribution"] > 0].head(2)
    top_neg = c[c["contribution"] < 0].head(1)
    feat_desc = "; ".join(
        f"{r.feature_ko} {r.value:,.1f} (학습 평균 대비 {r.z:+.1f}SD, 판단 기여 {r.contribution:+.2f})"
        for r in pd.concat([top_pos, top_neg]).itertuples()
    )
    bump_v = row["speed_bump_count"]
    record = {
        "case_type": key,
        "case_label": title,
        "selection_rule": rule_text,
        "region": row["sido_sgg_nm"],
        "ai_score_0_100": round(row["predicted_high_risk_probability_2026"] * 100, 1),
        "ai_rank_229": int(row["risk_rank_2026"]),
        "ai_rank_49": int(row["ai_rank_in_49"]),
        "infrastructure_score": round(row["infrastructure_score"], 3),
        "infrastructure_deficiency_score": round(row["infrastructure_deficiency_score"], 3),
        "deficiency_rank_49": int(row["deficiency_rank_in_49"]),
        "improvement_review_score": round(row["improvement_priority_score"], 3),
        "improvement_review_rank_49": int(row["improvement_priority_rank"]),
        "child_population_2025": int(row["child_population_0_12"]),
        "child_accidents_2025": int(row["prev_accident_count"]),
        "child_injuries_2025": int(row["prev_injury_count"]),
        "child_deaths_2025": int(row["prev_death_count"]),
        "accident_per_10k_child_2025": round(row["prev_accident_per_10k_child"], 1),
        "accident_count_mean_2y": round(row["accident_count_mean_2y"], 1),
        "child_accidents_2020_2025_total": int(row["child_accident_count"]),
        "vehicle_count_2025": int(row["prev_vehicle_count"]),
        "vehicles_per_100_child_2025": round(row["prev_vehicles_per_100_child"], 1),
        "schoolzone_count": fmt_int(row["schoolzone_count"]),
        "schoolzone_per_1000_child": round(row["schoolzone_per_1000_child"], 2),
        "crosswalk_count": fmt_int(row["crosswalk_count"]),
        "crosswalk_per_1000_child": round(row["crosswalk_per_1000_child"], 2),
        "traffic_light_count": fmt_int(row["traffic_light_count"]),
        "traffic_light_per_1000_child": round(row["traffic_light_per_1000_child"], 2),
        "cctv_installed_schoolzone_count": fmt_int(row["cctv_installed_zone_count"]),
        "cctv_install_rate_pct": round(row["cctv_install_rate"] * 100, 1),
        "speed_bump_count": fmt_int(bump_v),  # 35개 지역만 제공: 없으면 '데이터 미제공'(0 아님)
        "model_feature_highlights": feat_desc,
    }
    rows.append(record)

    supply = {
        "횡단보도(1천 명당)": pct49("crosswalk_per_1000_child", row["crosswalk_per_1000_child"]),
        "신호등(1천 명당)": pct49("traffic_light_per_1000_child", row["traffic_light_per_1000_child"]),
        "어린이보호구역(1천 명당)": pct49("schoolzone_per_1000_child", row["schoolzone_per_1000_child"]),
        "CCTV 설치율": pct49("cctv_install_rate", row["cctv_install_rate"]),
    }
    low = [k for k, v in supply.items() if v <= 40]
    high = [k for k, v in supply.items() if v >= 60]
    why = {
        "A": (f"AI 점수가 49개 지역 중 {record['ai_rank_49']}위(229개 중 {record['ai_rank_229']}위)로 높고, "
              f"시설 부족도도 49개 중 {record['deficiency_rank_49']}위로 높은 편이라 두 축이 함께 개선 검토 점수를 끌어올린 경우입니다."),
        "B": (f"AI 점수는 49개 지역 중 {record['ai_rank_49']}위(229개 중 {record['ai_rank_229']}위)로 가장 높은 수준이지만, "
              f"시설 부족도는 49개 중 {record['deficiency_rank_49']}위로 중간 이하입니다. 개선 검토 점수({record['improvement_review_rank_49']}위)는 주로 AI 점수에서 나온 경우입니다."),
        "C": (f"AI 점수는 49개 중 {record['ai_rank_49']}위(229개 중 {record['ai_rank_229']}위)로 중간 수준이지만, "
              f"시설 부족도가 49개 중 {record['deficiency_rank_49']}위로 높아 개선 검토 점수가 {record['improvement_review_rank_49']}위까지 올라온 경우입니다."),
    }[key]
    case_md.append(f"""### 사례 {key}. {row['sido_sgg_nm']} — {title}
- 선정 규칙: {rule_text}
- **AI 상대적 고위험 점수** {record['ai_score_0_100']:.1f}점 (229개 중 {record['ai_rank_229']}위 / 49개 중 {record['ai_rank_49']}위)
- **시설 부족도** {record['infrastructure_deficiency_score']:.3f} (49개 중 {record['deficiency_rank_49']}위) · **개선 검토 점수** {record['improvement_review_score']:.3f} (49개 중 {record['improvement_review_rank_49']}위)
- 사고(2025): 어린이 교통사고 {record['child_accidents_2025']}건, 부상자 {record['child_injuries_2025']}명, 사망자 {record['child_deaths_2025']}명, 어린이 1만 명당 사고율 {record['accident_per_10k_child_2025']:.1f} · 최근 2년 평균 {record['accident_count_mean_2y']:.1f}건 · 2020~2025 누적 {record['child_accidents_2020_2025_total']}건
- 어린이 인구(2025, 0~12세): {record['child_population_2025']:,}명 · 자동차 등록 {record['vehicle_count_2025']:,}대 (어린이 100명당 {record['vehicles_per_100_child_2025']:,.1f}대)
- 시설: 어린이보호구역 {record['schoolzone_count']}곳, 횡단보도 {record['crosswalk_count']}개, 신호등 {record['traffic_light_count']}개, CCTV 설치 보호구역 {record['cctv_installed_schoolzone_count']}곳(설치율 {record['cctv_install_rate_pct']:.1f}%), 과속방지턱 {record['speed_bump_count']}{'' if record['speed_bump_count'] == '데이터 미제공' else '개'}
- 49개 지역 내 시설 공급 백분위(높을수록 공급 많음): {', '.join(f'{k} {v:.0f}%' for k, v in supply.items())}
- 모델 입력 중 특징적인 값: {feat_desc}
- 왜 이 유형인가: {why} {('상대적으로 공급이 적은 항목: ' + ', '.join(low) + '.') if low else ''} {('상대적으로 공급이 많은 항목: ' + ', '.join(high) + '.') if high else ''}
- 활용: 시설이 사고 원인이라는 뜻이 아니라, AI 신호와 시설 공급 수준을 함께 볼 때 **현장 조사를 검토할 후보지역**이라는 의미입니다.
""")

cases = pd.DataFrame(rows)
cases.to_csv(OUT / "case_studies_3_regions.csv", index=False, encoding="utf-8-sig")

# 비교 그림: (왼쪽) 세 축 막대, (오른쪽) 시설 공급 백분위 레이더
fig = plt.figure(figsize=(13, 5.6))
ax1 = fig.add_subplot(1, 2, 1)
metrics_bar = ["AI 상대적 고위험 점수\n(0~1)", "시설 부족도\n(0~1)", "개선 검토 점수\n(0~1)"]
xb = np.arange(len(metrics_bar))
w = 0.26
for i, (key, row) in enumerate(picked.items()):
    vals = [row["predicted_high_risk_probability_2026"], row["infrastructure_deficiency_score"], row["improvement_priority_score"]]
    bars = ax1.bar(xb + (i - 1) * w, vals, w, color=CASE_COLORS[i], label=f"{key}. {row['sido_sgg_nm'].split()[-1]}")
    for b_, v in zip(bars, vals):
        ax1.text(b_.get_x() + b_.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.5, color=INK)
med = [ai.median(), dfc.median(), base["improvement_priority_score"].median()]
for xi, m in zip(xb, med):
    ax1.plot([xi - 1.5 * w, xi + 1.5 * w], [m, m], color=MUTED, lw=1.2, ls="--")
ax1.plot([], [], color=MUTED, ls="--", label="49개 지역 중앙값")
ax1.set_xticks(xb)
ax1.set_xticklabels(metrics_bar)
ax1.set_ylim(0, 1.12)
ax1.set_title("AI 위험도 · 시설 부족도 · 개선 검토 점수")
ax1.legend(frameon=False, fontsize=9, loc="upper right", ncol=2)
ax1.grid(axis="y", color=GRID)
ax1.set_axisbelow(True)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)

labels = ["횡단보도\n(1천 명당)", "신호등\n(1천 명당)", "어린이보호구역\n(1천 명당)", "CCTV\n설치율"]
cols = ["crosswalk_per_1000_child", "traffic_light_per_1000_child", "schoolzone_per_1000_child", "cctv_install_rate"]
ang = np.linspace(0, 2 * np.pi, len(cols), endpoint=False).tolist()
ang += ang[:1]
ax2 = fig.add_subplot(1, 2, 2, polar=True)
ax2.plot(ang, [50] * len(ang), color=MUTED, lw=1, ls="--")
for i, (key, row) in enumerate(picked.items()):
    vals = [pct49(c_, row[c_]) for c_ in cols]
    vals += vals[:1]
    ax2.plot(ang, vals, color=CASE_COLORS[i], lw=2, label=f"{key}. {row['sido_sgg_nm'].split()[-1]}")
    ax2.fill(ang, vals, color=CASE_COLORS[i], alpha=0.08)
ax2.set_xticks(ang[:-1])
ax2.set_xticklabels(labels, fontsize=9.5)
ax2.set_ylim(0, 100)
ax2.set_yticks([25, 50, 75, 100])
ax2.set_yticklabels(["25", "50", "75", "100"], fontsize=8, color=MUTED)
ax2.set_title("시설 공급 수준 (49개 지역 내 백분위, 바깥일수록 공급 많음)\n점선 = 중앙값", pad=18, fontsize=12)
ax2.legend(frameon=False, fontsize=9, loc="upper right", bbox_to_anchor=(1.32, 1.12))
fig.text(0.5, 0.01, "현장 조사 후보 선별용 분석 지표이며 시설과 사고의 인과관계를 뜻하지 않습니다. 시설 미제공 지역은 49개 분석 대상에서 제외(0으로 처리하지 않음).",
         ha="center", fontsize=8.5, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(OUT / "case_studies_comparison.png", dpi=180)
plt.close(fig)

compare = cases[["case_type", "region", "ai_score_0_100", "ai_rank_229", "infrastructure_deficiency_score",
                 "deficiency_rank_49", "improvement_review_score", "improvement_review_rank_49",
                 "child_population_2025", "child_accidents_2025", "accident_per_10k_child_2025",
                 "crosswalk_count", "traffic_light_count", "schoolzone_count", "cctv_install_rate_pct", "speed_bump_count"]].copy()
for col in ("child_population_2025", "child_accidents_2025"):
    compare[col] = compare[col].map(lambda v: f"{int(v):,}")
compare["ai_score_0_100"] = compare["ai_score_0_100"].map(lambda v: f"{v:.1f}점")
compare.columns = ["유형", "지역", "AI 점수", "AI 순위(229)", "시설 부족도", "부족도 순위(49)", "개선 검토 점수",
                   "개선 검토 순위(49)", "어린이 인구", "사고(2025)", "1만 명당 사고율", "횡단보도", "신호등",
                   "보호구역", "CCTV 설치율(%)", "과속방지턱"]
case_summary_md = f"""# 대표 지역 사례 3개 (시설 데이터가 있는 49개 지역)

> 사례는 미리 정한 규칙으로 49개 지역 안에서 선정했습니다(임의 선택 아님). 시설 부족과 사고 사이의 인과관계를 뜻하지 않으며, **현장 조사가 필요한 후보지역**을 설명하기 위한 자료입니다. 개선 검토 점수는 공식 정책 우선순위가 아닌 분석용 후보 점수입니다.

## 선정 기준 (49개 지역 분포 기준)
- AI 점수 25% / 75% 지점: {ai_q1:.3f} / {ai_q3:.3f} · 시설 부족도 중앙값 / 75% 지점: {def_med:.3f} / {def_q3:.3f}
- A: {rules['A'][1]} (조건 충족 상위: {candidates_note['A']})
- B: {rules['B'][1]} (조건 충족 상위: {candidates_note['B']})
- C: {rules['C'][1]} (조건 충족 상위: {candidates_note['C']})

## 개선 검토 상위 지역(정선군·태백시·태안군) 조건 확인
{chr(10).join(check_named)}

세 지역 모두 A 유형의 후보군(49개 중 AI 점수 상위 25%)에 들지만, A 사례는 규칙대로 그중 시설 부족도가 가장 높은 정선군을 사용했습니다. 태백시·태안군은 시설 부족도가 49개 중 {int(base.loc[base.sido_sgg_nm=='강원특별자치도 태백시','deficiency_rank_in_49'].iloc[0])}위, {int(base.loc[base.sido_sgg_nm=='충청남도 태안군','deficiency_rank_in_49'].iloc[0])}위로 중앙값보다 높아 B 유형(부족도 중앙값 미만)에는 해당하지 않고, AI 점수가 높아 C 유형(AI 중간)에도 해당하지 않습니다.

## 한눈에 비교
{md_table(compare)}

- 사고·인구·자동차 값은 2025년(2026 예측의 입력 연도) 기준입니다. 시설 수는 시설 분석 데이터 기준입니다.
- 과속방지턱은 35개 지역만 제공됩니다. 없는 지역은 '데이터 미제공'으로 표시하며 0으로 처리하지 않습니다.
- CCTV는 'CCTV가 설치된 어린이보호구역 수'이며 CCTV 대수가 아닙니다.

## 사례별 설명
{chr(10).join(case_md)}

## 사례를 함께 볼 때
- A와 B는 AI 점수가 비슷하게 높아도 시설 부족도에 따라 개선 검토 순위가 달라질 수 있음을 보여줍니다.
- C는 AI 점수만 보면 상위권이 아니지만 시설 공급 수준을 함께 보면 검토 대상에 들어오는 경우로, 두 지표를 결합한 이유를 설명합니다.
- '모델 입력 중 특징적인 값'의 판단 기여는 표준화 값 × 계수(log-odds)로, 그 지역 점수가 왜 높거나 낮게 나왔는지 보여주는 모델 내부 설명입니다(인과 아님).

산출물: `case_studies_3_regions.csv`, `case_studies_comparison.png`
"""
(OUT / "case_studies_summary.md").write_text(case_summary_md, encoding="utf-8")

# =====================================================================
# 3. 개선 검토 점수 가중치 민감도
# =====================================================================
# 현재 산식(노트북 14): score = (AI 점수 + (1 - infrastructure_score)) / 2  → AI 50% / 시설 부족도 50%
recomputed = (blind["predicted_high_risk_probability_2026"] + (1 - blind["infrastructure_score"])) / 2
formula_max_diff = float((recomputed - blind["improvement_priority_score"]).abs().max())
assert formula_max_diff < 1e-9, "현재 산식이 결과 파일과 일치하지 않음"

sens = blind[["sido_sgg_nm", "improvement_priority_rank", "predicted_high_risk_probability_2026",
              "infrastructure_deficiency_score", "improvement_priority_score"]].copy()
SCENARIOS = [(0.5, "기준안 AI50:시설50 (현재)"), (0.6, "AI60:시설40"), (0.7, "AI70:시설30"),
             (0.4, "AI40:시설60"), (0.3, "AI30:시설70")]


def ranks_for(w_ai: float) -> pd.Series:
    s_ = w_ai * sens["predicted_high_risk_probability_2026"] + (1 - w_ai) * sens["infrastructure_deficiency_score"]
    return s_, s_.rank(ascending=False, method="min").astype(int)


for w_ai, _ in SCENARIOS:
    tag = f"ai{int(w_ai * 100)}"
    sens[f"score_{tag}"], sens[f"rank_{tag}"] = ranks_for(w_ai)
assert (sens["rank_ai50"] == sens["improvement_priority_rank"]).all(), "기준안 순위가 기존 순위와 다름"

# 보조 점검: 두 축을 모두 49개 내 백분위로 바꿔 50:50 (척도 차이 영향 확인용, 기준안 아님)
ai_pct = sens["predicted_high_risk_probability_2026"].rank(pct=True)
def_pct = sens["infrastructure_deficiency_score"].rank(pct=True)
sens["score_pct50_supplementary"] = (ai_pct + def_pct) / 2
sens["rank_pct50_supplementary"] = sens["score_pct50_supplementary"].rank(ascending=False, method="min").astype(int)

base_rank = sens["rank_ai50"]
top10_base = set(sens.loc[base_rank <= 10, "sido_sgg_nm"])
top5_base = set(sens.loc[base_rank <= 5, "sido_sgg_nm"])
summary_rows = []
scenario_cols = [(f"rank_ai{int(w * 100)}", label, w) for w, label in SCENARIOS] + [
    ("rank_pct50_supplementary", "보조: 두 축 백분위 50:50", None)]
for col, label, w in scenario_cols:
    r_ = sens[col]
    rho = spearmanr(base_rank, r_).statistic
    tau = kendalltau(base_rank, r_).statistic
    top10 = set(sens.loc[r_ <= 10, "sido_sgg_nm"])
    top5 = set(sens.loc[r_ <= 5, "sido_sgg_nm"])
    shift = (r_ - base_rank).abs()
    big = sens.loc[shift.sort_values(ascending=False).index[:3]]
    summary_rows.append({
        "scenario": label,
        "ai_weight": "" if w is None else w,
        "facility_deficiency_weight": "" if w is None else round(1 - w, 1),
        "spearman_rho_vs_baseline": round(float(rho), 4),
        "kendall_tau_vs_baseline": round(float(tau), 4),
        "top10_common_with_baseline": len(top10 & top10_base),
        "top5_common_with_baseline": len(top5 & top5_base),
        "mean_abs_rank_change": round(float(shift.mean()), 2),
        "max_abs_rank_change": int(shift.max()),
        "largest_movers": "; ".join(f"{n.split()[-1]} {int(a)}→{int(b_)}" for n, a, b_ in
                                    zip(big["sido_sgg_nm"], base_rank[big.index], r_[big.index])),
        "top10_regions": ", ".join(n.split()[-1] for n in sens.loc[r_.sort_values().index[:10], "sido_sgg_nm"]),
    })
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / "weight_sensitivity.csv", index=False, encoding="utf-8-sig")

main_rank_cols = [c for c, _, w in scenario_cols if w is not None]
sens["best_rank"] = sens[main_rank_cols].min(axis=1)
sens["worst_rank"] = sens[main_rank_cols].max(axis=1)
sens["rank_range"] = sens["worst_rank"] - sens["best_rank"]
sens["in_top10_all_5_scenarios"] = (sens[main_rank_cols] <= 10).all(axis=1)
sens["in_top5_all_5_scenarios"] = (sens[main_rank_cols] <= 5).all(axis=1)
sens["times_in_top10"] = (sens[main_rank_cols] <= 10).sum(axis=1)
sens.sort_values("improvement_priority_rank").to_csv(OUT / "weight_sensitivity_by_region.csv", index=False, encoding="utf-8-sig")

top_cmp = pd.DataFrame({"rank": range(1, 11)})
for col, label, w in scenario_cols:
    order = sens.loc[sens[col].sort_values(kind="stable").index, "sido_sgg_nm"].head(10).tolist()
    top_cmp[label] = order
top_cmp.to_csv(OUT / "weight_top10_comparison.csv", index=False, encoding="utf-8-sig")

stable10 = sens[sens["in_top10_all_5_scenarios"]].sort_values("improvement_priority_rank")
stable5 = sens[sens["in_top5_all_5_scenarios"]].sort_values("improvement_priority_rank")
volatile = sens[sens["times_in_top10"].between(1, 4)].sort_values("rank_range", ascending=False)

# 연속 가중치 곡선(0.1 간격)
grid = np.round(np.arange(0.0, 1.01, 0.1), 1)
curve = []
for w_ai in grid:
    _, r_ = ranks_for(w_ai)
    curve.append({
        "w": w_ai,
        "rho": spearmanr(base_rank, r_).statistic,
        "top10": len(set(sens.loc[r_ <= 10, "sido_sgg_nm"]) & top10_base),
        "top5": len(set(sens.loc[r_ <= 5, "sido_sgg_nm"]) & top5_base),
    })
curve = pd.DataFrame(curve)

# 그림: (왼쪽) 상위 후보 순위 heatmap, (오른쪽) 가중치별 안정성 곡선
union = sens[(sens[main_rank_cols] <= 10).any(axis=1)].sort_values("improvement_priority_rank")
heat = union[main_rank_cols].to_numpy()
fig = plt.figure(figsize=(14, 7.2))
ax1 = fig.add_axes([0.14, 0.12, 0.37, 0.78])
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("rank", ["#0f6e5e", "#5fbfa9", "#e9f5f2", "#ffffff"])
norm = matplotlib.colors.Normalize(vmin=1, vmax=25, clip=True)
ax1.imshow(heat, cmap=cmap, norm=norm, aspect="auto")
for i in range(heat.shape[0]):
    for j in range(heat.shape[1]):
        v = int(heat[i, j])
        ax1.text(j, i, v, ha="center", va="center", fontsize=9,
                 color="white" if v <= 8 else INK, fontweight="bold" if v <= 10 else "normal")
ax1.set_xticks(range(len(main_rank_cols)))
ax1.set_xticklabels([label.replace(" (현재)", "\n(현재)").replace("기준안 ", "") for _, label, w in scenario_cols if w is not None], fontsize=9)
ax1.set_yticks(range(len(union)))
ax1.set_yticklabels([f"{n}{'  ★' if s else ''}" for n, s in zip(union["sido_sgg_nm"], union["in_top10_all_5_scenarios"])], fontsize=9)
ax1.set_title("가중치별 개선 검토 순위 (어느 시나리오에서든 TOP10에 든 지역)\n★ = 5개 시나리오 모두 TOP10 유지", fontsize=12)
ax1.tick_params(length=0)
for s in ax1.spines.values():
    s.set_visible(False)

ax2 = fig.add_axes([0.6, 0.12, 0.34, 0.78])
ax2.plot(curve["w"], curve["top10"], color=TEAL, lw=2.2, marker="o", label="기준안과 공통 TOP10 수 (왼쪽 축)")
ax2.plot(curve["w"], curve["top5"], color="#e3a21a", lw=2.2, marker="s", label="기준안과 공통 TOP5 수 (왼쪽 축)")
ax2.set_ylim(0, 10.8)
ax2.set_xlabel("AI 상대적 고위험 점수 가중치 (나머지는 시설 부족도)")
ax2.set_ylabel("공통 지역 수")
ax2.axvspan(0.3, 0.7, color="#f1f3f5", zorder=0)
ax2.axvline(0.5, color=INK, lw=1, ls="--")
ax2.text(0.51, 1.1, "현재 50:50", fontsize=9, color=INK)
ax2.text(0.31, 0.3, "비교 구간 30~70%", fontsize=8.5, color=MUTED)
ax3 = ax2.twinx()
ax3.plot(curve["w"], curve["rho"], color="#6f42c1", lw=1.8, ls=":", marker="^", label="Spearman ρ (오른쪽 축)")
ax3.set_ylim(0, 1.08)
ax3.set_ylabel("기준안과의 Spearman 순위상관")
h1, l1 = ax2.get_legend_handles_labels()
h2, l2 = ax3.get_legend_handles_labels()
ax2.legend(h1 + h2, l1 + l2, frameon=False, fontsize=9, loc="upper left")
ax2.set_title("가중치 변화에 따른 상위 후보 안정성 (0.1 간격)", fontsize=12)
for s in ("top",):
    ax2.spines[s].set_visible(False)
    ax3.spines[s].set_visible(False)
fig.text(0.5, 0.02, "개선 검토 점수 = w × AI 상대적 고위험 점수 + (1 − w) × 시설 부족도. 분석용 후보 점수이며 공식 정책 우선순위가 아님.",
         ha="center", fontsize=9, color=MUTED)
fig.savefig(OUT / "weight_sensitivity.png", dpi=180)
plt.close(fig)

main_summary = summary.iloc[:5]
min_rho = main_summary["spearman_rho_vs_baseline"].min()
min_top10 = main_summary["top10_common_with_baseline"].min()
min_top5 = main_summary["top5_common_with_baseline"].min()
near = main_summary[main_summary["ai_weight"].isin([0.4, 0.6])]
far = main_summary[main_summary["ai_weight"].isin([0.3, 0.7])]
stable_phrase = (
    "가중치 변화에도 상위 후보가 비교적 안정적"
    if near["top10_common_with_baseline"].min() >= 8 and near["spearman_rho_vs_baseline"].min() >= 0.9
    else "가중치 선택에 민감하므로 정책 활용 전 추가 검증 필요"
)
far_phrase = (
    "30:70·70:30처럼 크게 바꿔도 상위권 대부분이 유지됩니다."
    if far["top10_common_with_baseline"].min() >= 8
    else f"다만 30:70·70:30처럼 크게 바꾸면 TOP10 중 최소 {int(far['top10_common_with_baseline'].min())}개만 유지되어, "
         "가중치 선택에 민감한 구간이 있으므로 정책 활용 전 추가 검증이 필요합니다."
)
sup = summary.iloc[5]
spread_ai, spread_def = float(sens["predicted_high_risk_probability_2026"].std()), float(sens["infrastructure_deficiency_score"].std())

tbl = main_summary[["scenario", "spearman_rho_vs_baseline", "top10_common_with_baseline", "top5_common_with_baseline",
                    "mean_abs_rank_change", "max_abs_rank_change", "largest_movers"]].copy()
tbl.columns = ["시나리오", "Spearman ρ", "공통 TOP10", "공통 TOP5", "평균 순위 변동", "최대 순위 변동", "변동 큰 지역(기준→시나리오)"]
top_tbl = top_cmp[["rank"] + [label for _, label, w in scenario_cols if w is not None]].copy()
for c in top_tbl.columns[1:]:
    top_tbl[c] = top_tbl[c].map(lambda n: n.split()[-1])
top_tbl.columns = ["순위"] + [c.replace("기준안 ", "") for c in top_tbl.columns[1:]]

weight_md = f"""# 개선 검토 점수 가중치 민감도 분석 (49개 시설분석 지역)

> 개선 검토 점수는 공식 정책 우선순위가 아니라 현장점검 후보를 고르기 위한 **분석용 후보 점수**입니다.

## 1. 현재 실제 산식 (코드 확인)
- 출처: `notebook/14_인프라_사각지대_도출.ipynb` → `results/infrastructure_blind_spots_49.csv`
- 시설 부족도 = 1 − infrastructure_score
  - infrastructure_score(노트북 10) = 49개 지역 내 백분위 점수 4개의 평균: 어린이 1천 명당 횡단보도, 1천 명당 신호등, 1천 명당 어린이보호구역, CCTV 설치율
- **개선 검토 점수 = (AI 상대적 고위험 점수 + 시설 부족도) / 2 → AI 50% : 시설 부족도 50% (동일 가중치)**
- 검증: 이 산식으로 다시 계산한 값과 결과 파일 값의 최대 차이 {formula_max_diff:.1e}, 순위 49개 모두 동일

## 2. 비교 시나리오
AI 50:시설 50(기준안, 현재) / 60:40 / 70:30 / 40:60 / 30:70. 각 시나리오에서 49개 지역의 점수와 순위(동점은 같은 순위)를 다시 계산했습니다. 기존 결과 파일은 바꾸지 않았습니다.

## 3. 결과
{md_table(tbl)}

### 가중치별 TOP10
{md_table(top_tbl)}

### 안정적 후보 (5개 시나리오 모두 TOP10)
{', '.join(f"{n} (기준 {int(r)}위, 범위 {int(b_)}~{int(w_)}위)" for n, r, b_, w_ in zip(stable10['sido_sgg_nm'], stable10['improvement_priority_rank'], stable10['best_rank'], stable10['worst_rank'])) or '없음'}

- 5개 시나리오 모두 TOP5: {', '.join(stable5['sido_sgg_nm']) or '없음'}

### 순위 민감도가 큰 후보 (일부 시나리오에서만 TOP10)
{chr(10).join(f"- {r.sido_sgg_nm}: 기준 {int(r.improvement_priority_rank)}위, 시나리오별 {int(r.best_rank)}~{int(r.worst_rank)}위 (TOP10 {int(r.times_in_top10)}/5회)" for r in volatile.itertuples()) or '- 없음'}

## 4. 해석
- 기준안과 각 시나리오의 Spearman 순위상관은 최소 {min_rho:.3f}, 공통 TOP10은 최소 {int(min_top10)}개, 공통 TOP5는 최소 {int(min_top5)}개입니다.
- 40:60·60:40 범위: **{stable_phrase}**. {far_phrase}
- 상위권에서 순위가 흔들리는 지역은 대개 AI 점수와 시설 부족도 중 한쪽만 높은 지역입니다(가중치가 어느 쪽에 실리느냐에 따라 오르내림). 이런 지역은 두 축을 따로 보여주며 설명하는 것이 안전합니다.

## 5. 가중치 설정의 한계
- 50:50은 "두 신호를 같은 비중으로 본다"는 설명 가능한 휴리스틱이며, 데이터로 최적화한 값이 아닙니다(정답 가중치를 정할 기준 자료가 없음).
- 두 축의 척도가 다릅니다: 49개 지역에서 AI 점수 표준편차 {spread_ai:.3f}, 시설 부족도 표준편차 {spread_def:.3f}. AI 점수의 퍼짐이 더 커서 같은 50:50이어도 순위에는 AI 점수가 더 크게 작용합니다.
  - 보조 점검: 두 축을 모두 49개 내 백분위로 바꿔 50:50으로 계산하면 기준안과 Spearman ρ {sup['spearman_rho_vs_baseline']:.3f}, 공통 TOP10 {int(sup['top10_common_with_baseline'])}개, 공통 TOP5 {int(sup['top5_common_with_baseline'])}개입니다(참고용, 기준안 아님).
- 시설 부족도는 시설 데이터가 확보된 49개 지역 안에서의 상대 백분위입니다. 49개 밖 지역과는 비교할 수 없고, 시설이 없는 것으로 간주하지 않습니다.
- 시설 '수'는 공급량 지표일 뿐 위치·상태·실효성은 반영하지 않으므로, 상위 후보는 현장 확인이 필요합니다.

산출물: `weight_sensitivity.csv`(시나리오 요약), `weight_sensitivity_by_region.csv`(지역별 점수·순위), `weight_top10_comparison.csv`, `weight_sensitivity.png`
"""
(OUT / "weight_sensitivity_summary.md").write_text(weight_md, encoding="utf-8")

# =====================================================================
# 4. 종합 요약
# =====================================================================
top5_feat = expl.head(5)
case_lines = "\n".join(
    f"- **{r['case_type']}. {r['region']}** ({r['case_label']}): AI {r['ai_score_0_100']:.1f}점(229개 중 {r['ai_rank_229']}위), "
    f"시설 부족도 {r['infrastructure_deficiency_score']:.3f}(49개 중 {r['deficiency_rank_49']}위), 개선 검토 {r['improvement_review_rank_49']}위"
    for r in rows
)
overall = f"""# SaveSpot 공모전 설명용 분석 요약

> 이 문서의 모든 결과는 기존 모델·2026년 229개 예측·개선 검토 결과를 **읽기만 해서** 만든 설명 자료입니다. AI 점수는 실제 사고 발생 확률이 아니라 지역 간 상대적 고위험군 분류 점수이고, 개선 검토 점수는 공식 정책 우선순위가 아닌 분석용 후보 점수입니다.

## 1. 모델 설명 가능성
- **Logistic Regression 선택 이유:** 2023~2025 연도별 순차 검증(walk-forward) 평균 ROC-AUC {lr_m['roc_auc']:.3f}, PR-AUC {lr_m['average_precision']:.3f}, Recall {lr_m['recall']:.3f}로 비교 모델(Random Forest, HistGradientBoosting)보다 높았고, 계수로 판단 방향을 설명할 수 있습니다.
- **주요 영향 변수(표준화 계수 절댓값 TOP5):**
{chr(10).join(f'  {i + 1}. {r.feature_ko} ({r.coefficient_standardized:+.2f}, {r.direction})' for i, r in enumerate(top5_feat.itertuples()))}
- **방향:** 사고 수준 변수(전년 사고 건수, 최근 2년 평균, 전년 사고율, 부상자 수)가 높을수록 고위험군 방향, 어린이 인구가 많을수록(같은 사고 수준이면 사고율이 낮아지는 구조) 저위험군 방향입니다. 자동차 수·사망자 수의 영향은 작습니다.
- **주의:** 상관/모델 판단 방향이지 인과관계가 아닙니다. 사고 관련 변수끼리 상관이 높아 개별 계수는 묶어서 해석합니다. 자세한 내용: `model_explainability_summary.md`

## 2. 대표 지역 사례 3개 (49개 시설분석 지역, 규칙 기반 선정)
{case_lines}
- **차이:** A는 두 축이 함께 높아 상위 후보가 된 경우, B는 AI 신호는 가장 강하지만 시설 공급은 중간 이상인 경우, C는 AI 신호는 중간이지만 시설 부족도가 높아 후보로 올라온 경우입니다. 같은 '후보'라도 확인할 내용이 다릅니다(A: 종합 점검, B: 시설 외 요인 확인, C: 시설 공급 점검). 자세한 내용: `case_studies_summary.md`

## 3. 가중치 민감도 분석
- **현재 산식:** 개선 검토 점수 = (AI 상대적 고위험 점수 + 시설 부족도) / 2 (AI 50 : 시설 50, 노트북 14에서 확인)
- **비교 가중치:** 60:40, 70:30, 40:60, 30:70
- **순위 안정성:** 기준안과 Spearman ρ 최소 {min_rho:.3f}, 공통 TOP10 최소 {int(min_top10)}개, 공통 TOP5 최소 {int(min_top5)}개
- **상위 후보 유지:** 5개 시나리오 모두 TOP10 — {', '.join(n.split()[-1] for n in stable10['sido_sgg_nm']) or '없음'}
- **판단:** 40:60~60:40 범위에서는 {stable_phrase}. {far_phrase}
- **한계:** 50:50은 최적화 값이 아닌 설명 가능한 동일 가중치이며, 두 축의 분포 폭이 달라 실제로는 AI 점수가 순위에 더 크게 작용합니다. 자세한 내용: `weight_sensitivity_summary.md`

## 4. 공모전 발표용 핵심 문장
1. SaveSpot의 AI는 사고 발생 확률을 예측한 것이 아니라, 과거 사고·인구 자료로 229개 시군구 중 **상대적 고위험 지역을 분류**합니다.
2. 모델 계수를 보면 최근 사고 수준이 높은 지역일수록 고위험 방향으로 판단하며, 이는 원인 분석이 아니라 모델의 판단 근거를 투명하게 공개한 것입니다.
3. 시설 데이터는 모델 입력이 아니라 별도의 시설 부족도 분석에 사용했고, 두 결과는 시설 데이터가 확보된 **49개 지역에서만** 결합했습니다.
4. 개선 검토 점수는 공식 정책 우선순위가 아니라 **현장점검 후보를 선별하기 위한 분석 지표**이며, {'·'.join(r['region'].split()[-1] for r in rows)}처럼 후보가 된 이유가 서로 다르다는 점까지 함께 제시합니다.
5. 가중치를 40:60~60:40으로 바꿔도 상위 후보 대부분이 유지되는지 민감도 분석으로 확인했고, 크게 바꿨을 때 흔들리는 지역도 숨기지 않고 공개했습니다.

## 5. 주의할 점
- AI 점수·계수를 사고 확률이나 인과 효과로 표현하지 않습니다.
- 시설 부족도는 49개 지역 안의 상대 비교이며, 나머지 180개 지역은 시설 부족 여부를 판단하지 않습니다(0으로 보지 않음).
- 과속방지턱은 35개 지역만 제공되어 개선 검토 점수에 들어가지 않습니다.
- 시설 수는 위치·상태·실효성을 반영하지 않으므로 상위 후보는 현장 확인이 필요합니다.

## 산출물 목록 (`results/competition/`)
- `model_explainability.csv` / `.png` / `_direction.png` / `_summary.md`
- `case_studies_3_regions.csv` / `case_studies_comparison.png` / `case_studies_summary.md`
- `weight_sensitivity.csv` / `weight_sensitivity_by_region.csv` / `weight_top10_comparison.csv` / `weight_sensitivity.png` / `weight_sensitivity_summary.md`
- 재생성: `python scripts/competition_analysis.py`
"""
(OUT / "competition_analysis_summary.md").write_text(overall, encoding="utf-8")

# ---------------------------------------------------------------- 보존 확인
INPUT_HASHES_AFTER = {p.name: sha256(p) for p in (PRED_PATH, BLIND_PATH, MODEL_PATH)}
assert INPUT_HASHES_BEFORE == INPUT_HASHES_AFTER, "입력 파일이 변경됨"
print("model reproduce max diff:", max_repro_diff)
print("formula max diff:", formula_max_diff)
print("cases:", [r["region"] for r in rows])
print(summary[["scenario", "spearman_rho_vs_baseline", "top10_common_with_baseline", "top5_common_with_baseline", "max_abs_rank_change"]].to_string())
print("stable top10:", list(stable10["sido_sgg_nm"]))
print("volatile:", list(volatile["sido_sgg_nm"]))
print("inputs unchanged:", INPUT_HASHES_BEFORE == INPUT_HASHES_AFTER)
