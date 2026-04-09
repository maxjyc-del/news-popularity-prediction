# Data

학회에서 제공한 분할 데이터는 레포에 포함하지 않습니다.

## 원본 데이터
- **UCI Machine Learning Repository — Online News Popularity**
  - https://archive.ics.uci.edu/dataset/332/online+news+popularity
  - Mashable 기사 39,644건, 라이선스 CC BY 4.0
  - Fernandes, K., Vinagre, P., Cortez, P., & Sernadela, P. (2015). *Online News Popularity* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5NS3V

학회 데이터는 원본에서 `url`, `timedelta`, `shares`를 제거하고 이진 타깃 `is_popular`를 붙여
train / leaderboard / test로 나눈 형태입니다.

## 배치 방법
노트북을 실행하려면 아래 파일명으로 이 폴더에 넣으세요.

```
data/
├── train.csv                          # id + 피처 58개 + is_popular (23,644행)
├── leaderboard.csv                    # id + 피처 58개 (8,000행)
├── test.csv                           # id + 피처 58개 (8,000행)
└── sample_submission_leaderboard.csv  # id, is_popular (8,000행)
```
