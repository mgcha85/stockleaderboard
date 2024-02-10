import pandas as pd

# 상승 추세 확인 함수
def is_upward_trend(df, index):
    if index < 60:
        return False
    return df['Close'][index] >= 1.3 * df['Close'][index - 20] and df['Close'][index] <= 2 * df['Close'][index - 60]

# 수렴 확인 함수
def is_consolidating(df, index):
    if index < 80:
        return False
    recent_range = max(df['High'][index-40:index]) - min(df['Low'][index-40:index])
    past_range = max(df['High'][index-80:index-40]) - min(df['Low'][index-80:index-40])
    return recent_range < 0.5 * past_range

# EMA 정배열 확인 함수
def is_ema_aligned(df, index):
    return df['EMA10'][index] > df['EMA20'][index] and df['EMA20'][index] > df['EMA50'][index]

# EMA와 캔들 위치 확인 함수
def is_ema50_below_candles(df, index):
    return all(df['Low'][i] > df['EMA50'][i] for i in range(index-50, index))

# EMA 계산을 위한 함수 추가
def calculate_ema(df):
    df['EMA10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()

# 메인 데이터 프레임 준비 (예시: 'df'라는 이름의 DataFrame)
# df = pd.read_csv('your_stock_data.csv')  # 주식 데이터 불러오기
# calculate_ema(df)  # EMA 계산

# 특정 인덱스에서 조건들을 모두 만족하는지 확인
def check_all_conditions(df, index):
    if is_upward_trend(df, index) and \
       is_consolidating(df, index) and \
       is_ema_aligned(df, index) and \
       is_ema50_below_candles(df, index):
        return True
    return False

# 전체 데이터 프레임에 대해 조건 확인
for i in range(len(df)):
    if check_all_conditions(df, i):
        # 조건을 만족하는 인덱스 출력 (예: 매수 신호)
        print(f"Condition met at index: {i}")
