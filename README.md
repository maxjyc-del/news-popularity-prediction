# 뉴스 기사 인기 예측 — LightGBM · XGBoost · CatBoost 앙상블

> 기사 메타데이터(본문 길이, 이미지·링크 수, 키워드 점수, 토픽, 감성, 게시 요일 등) 58개 피처로
> **인기 기사 여부(`is_popular`)** 의 확률을 예측하는 이진 분류 프로젝트.
> Optuna로 튜닝한 3개 부스팅 모델을 5-Fold × 2 Seed로 앙상블했습니다.

| 항목 | 내용 |
|---|---|
| 소속 | 데이터 사이언스 학회 Parrot 14기 — ML 프로젝트 |
| 기간 | <!-- TODO: 시작일 --> ~ 2026.04.09 (최종 제출·발표) |
| 평가 | PR-AUC (정량 70%, Private 기준) + 발표 (정성 30%) |
| 결과 | **Public PR-AUC 0.338**  |
| 스택 | Python, pandas, scikit-learn, LightGBM, XGBoost, CatBoost, Optuna |


---

## 1. 데이터

| 파일 | 규모 | 설명 |
|---|---|---|
| `train.csv` | 23,644행 | id + 피처 58개 + `is_popular` |
| `leaderboard.csv` | 8,000행 | Public 점수용 (정답 비공개) |
| `test.csv` | 8,000행 | Private 점수용 (정답 비공개) |

- 원본: [UCI Online News Popularity](https://archive.ics.uci.edu/dataset/332/online+news+popularity) (Mashable 기사 39,644건, CC BY 4.0)를 학회에서 3개 세트로 분할하고, 원본의 공유 수(`shares`) 대신 이진 타깃 `is_popular`로 제공
- 타깃 양성 비율 **약 16.8%** → 클래스 불균형 (음성:양성 ≈ 4.95:1)
- 데이터 파일은 레포에 포함하지 않았습니다 → [`data/README.md`](data/README.md)

## 2. 전처리

| 항목 | 내용 | 결정 |
|---|---|---|
| 결측치 | 없음 | 처리 불필요 |
| 상수 컬럼 | 없음 | 코드에서 자동 탐지·제거하도록 구성 |
| 이상치 | 비율형 피처(`n_unique_tokens`, `n_non_stop_words`, `n_non_stop_unique_tokens`)가 정상 범위 0~1을 벗어나 최대 700 이상 | `clip(0, 1)`, IQR 처리 시도 → **원본 유지** |
| 변환 | 스케일이 큰 피처 log 변환, `kw_avg_avg`·`kw_max_avg` 분위수 변환 | 둘 다 0.335로 하락 → **원본 유지 (0.338)** |

**원본을 그대로 쓴 이유**
1. 트리 모델은 값의 크기가 아니라 대소 관계로 분기하므로 스케일 변환의 효과가 거의 없음
2. `kw_avg_avg`, `LDA_*` 등 상위 피처가 이미 계산된 파생 피처라 추가 변환이 오히려 정보를 손실시킴
3. 분위수 변환 시 일부 피처의 타깃 상관은 올랐지만(0.088 → 0.172) `num_imgs`(0.079 → 0.011), `num_hrefs`(0.073 → 0.021)처럼 떨어지는 피처가 더 많았음

## 3. 모델링

### 3-1. 모델 선택 과정 (Public PR-AUC)

| 단계 | 모델 | PR-AUC | 비고 |
|---|---|---|---|
| 1 | Random Forest | 0.310 | 과적합, 성능 한계 |
| 2 | XGBoost 단일 | 0.327 | 부스팅으로 교체 |
| 3 | LightGBM + XGBoost | 0.337 | 서로 약점 보완 |
| 4 | + CatBoost | 0.33710 | 학습 방식 다양화 |
| 5 | + Optuna 튜닝 | 0.33756 | |
| 6 | 5 + 10 Fold 블렌딩 | 0.33761 | |
| 7 | **Optuna + Seed 블렌딩 (최종)** | **0.338** | |

### 3-2. 최종 파이프라인 ([`notebooks/news_popularity_ensemble.ipynb`](notebooks/news_popularity_ensemble.ipynb))

```
train (58 features)
 ├─ Optuna 튜닝 (3-Fold, PR-AUC 최대화)
 │    ├─ LightGBM 20 trials
 │    └─ CatBoost 15 trials           (XGBoost는 고정 파라미터)
 │
 ├─ Seed 42  : 5-Fold × {LGB, XGB, CAT} → OOF로 3모델 가중치 최적화 (Nelder-Mead)
 ├─ Seed 777 : 5-Fold × {LGB, XGB, CAT} → OOF로 3모델 가중치 최적화
 │
 └─ 두 Seed 결과를 OOF 기준 최적 비율로 블렌딩 → is_popular 확률
```

| | LightGBM | XGBoost | CatBoost |
|---|---|---|---|
| 트리 성장 | Leaf-wise | Level-wise | Symmetric |
| 불균형 처리 | `is_unbalance=True` | `scale_pos_weight=4.95` | `auto_class_weights='Balanced'` |
| 평가 지표 | `average_precision` | `aucpr` | `PRAUC` |
| 튜닝 | Optuna (lr, num_leaves, min_child_samples, subsample, colsample, L1/L2) | 고정 (lr 0.03, depth 6) | Optuna (lr, depth, l2_leaf_reg, bagging_temperature, random_strength) |
| Early stopping | 100 | 100 | 100 |

**Optuna 탐색 전략**
- **탐색 범위 축소**: learning_rate 0.01~0.1 → 0.01~0.05로 좁혀 이전 최적 파라미터 주변을 정밀 탐색
- **`enqueue_trial`로 시작점 고정**: 이전 최고 점수(0.337) 파라미터를 첫 trial로 넣어 처음부터 최적점 근처에서 탐색

**OOF 결과 (최종 실행)**

| | Seed 42 | Seed 777 | 블렌딩 |
|---|---|---|---|
| OOF PR-AUC | 0.33134 | 0.33012 | **0.33321** |

## 4. 결과 해석

### 타깃과 상관이 큰 피처

| 피처 | 의미 | 상관계수 | 방향 |
|---|---|---|---|
| `kw_avg_avg` | 키워드 평균 점수 | +0.170 | ↑ 인기 |
| `LDA_03` | 토픽 3 확률 | +0.118 | ↑ 인기 |
| `LDA_02` | 토픽 2 확률 | −0.110 | ↓ 비인기 |
| `data_channel_is_world` | 세계 뉴스 여부 | −0.095 | ↓ 비인기 |
| `kw_max_avg` | 키워드 최대 평균 | +0.088 | ↑ 인기 |
| `num_imgs` | 이미지 수 | +0.079 | ↑ 인기 |
| `num_hrefs` | 링크 수 | +0.073 | ↑ 인기 |
| `is_weekend` | 주말 게시 여부 | +0.067 | ↑ 인기 |

### 인기 기사 패턴
-  **키워드 품질이 핵심**: `kw_avg_avg` 상위 20% 기사의 인기율 29.6% vs 하위 20% 9.6% → 약 3배
-  **주말 효과**: 주말 인기율 23.2% vs 평일 15.8% → 약 1.5배
-  **채널 차이**: 소셜미디어(24%)·라이프스타일(21%)은 높고, 세계 뉴스(10%)는 가장 낮음
-  **풍부한 콘텐츠**: 이미지·링크가 많을수록 인기 비율이 높음

## 5. 어려웠던 점

- **전처리가 점수를 올리지 못함**: 변환을 시도할수록 점수가 떨어짐 → 트리 모델의 특성을 공부한 뒤 원본 사용으로 결정
- **학습 시간 증가**: 3모델 × 5-Fold × 2 Seed + Optuna로 실행 시간이 길어짐 → Early Stopping 도입, Optuna 탐색 범위 축소 (최종 약 35분)
- **피처 엔지니어링 효과 미미**: 파생 피처를 추가했지만 히트맵 분석에서 원본 대비 차이가 없어 제외
  
### 추가 개선 방향
- **가중치 제약**: Nelder-Mead 가중치 최적화에 범위 제약이 없어 Seed 777에서 XGB 가중치가 음수(−0.022)로 나옴 → 가중치를 0 이상으로 제한(SLSQP + bounds 등)할 필요
- **검증 분리**: 블렌딩 가중치를 평가에 쓰는 OOF와 같은 데이터로 최적화하므로 OOF 점수가 다소 낙관적일 수 있음 → 가중치 결정용 fold를 따로 두는 방식 고려
- **XGBoost 튜닝**: XGB만 고정 파라미터를 사용 → 함께 튜닝하면 앙상블 다양성과 성능 개선 여지

## 6. 실행 방법

```bash
pip install -r requirements.txt
# data/ 에 데이터 배치 (data/README.md 참고) 후
jupyter notebook notebooks/news_popularity_ensemble.ipynb
```
결과 파일은 `output/leaderboard_submission.csv`, `output/test_submission.csv`로 저장됩니다.

## 7. 레포 구조

```
├── notebooks/
│   └── news_popularity_ensemble.ipynb   # 최종 제출 코드 (v15, 실행 결과 포함)
├── docs/
│   └── presentation.pdf                 # 최종 발표 자료
├── data/README.md                       # 데이터 출처 및 배치 방법
└── requirements.txt
```
