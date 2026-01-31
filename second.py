import pandas as pd
import numpy as np
import yfinance as yf


# 1. 일반화된 EMA 함수(GEMA)
def generalized_ema(series, n, alpha):
    if len(series) < n:
        return pd.Series([np.nan] * len(series))

    def calc_weighted_avg(window_slice):
        # 최신 데이터에 가중치 1, 과거로 갈수록 alpha^i 적용
        # alpha ∈ (0,1), 작을수록 최근값 강조
        weights = np.array([(1 - alpha) * (alpha**i) for i in range(len(window_slice))])
        return np.sum(window_slice[::-1] * weights) / np.sum(weights)

    return series.rolling(window=n).apply(calc_weighted_avg, raw=True)


# 2. 크로스 엔트로피 계산 (MACD 시그널의 예측력 측정)
def calculate_cross_entropy(y_true, macd_line, signal_line):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # 실제값: 주가의 다음날 방향성 (오르면 1, 내리면 0에 가깝게)
    true_direction = sigmoid(y_true.pct_change().shift(-1).fillna(0))


    # 예측값: MACD Line이 Signal Line보다 위에 있는지 여부 (강세/약세)
    # 두 선의 차이(Oscillator)를 확률로 변환
    diff = (macd_line - signal_line).fillna(0)

    eps = 1e-12
    mu = diff.mean()
    s  = diff.std(ddof=0) + eps
    z  = (diff - mu) / s
    pred_prob = sigmoid(z)

    epsilon = 1e-15
    ce = -(
        true_direction * np.log(pred_prob + epsilon)
        + (1 - true_direction) * np.log(1 - pred_prob + epsilon)
    )
    return ce.mean()


# 3. 최적 파라미터(n1, n2, n3, alpha) 추출 엔진
def find_optimal_macd_full_params(ticker="SPY", period="3y"):
    print(f"--- {ticker} 데이터 분석 및 전처리 시작 ---")
    df = yf.download(ticker, period=period)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 파라미터 그리드 설정
    n1_grid = [10, 12, 14, 16]  # 단기
    n2_grid = [22, 26, 30, 34]  # 장기
    n3_grid = [7, 9, 11, 13]  # 시그널
    alpha_grid = [0.1, 0.3, 0.5, 0.7]  # 가중치

    best_ce = float("inf")
    best_params = {}

    total_combinations = len(n1_grid) * len(n2_grid) * len(n3_grid) * len(alpha_grid)
    print(f"탐색 시작 (총 조합 수: {total_combinations}개)")

    for n1 in n1_grid:
        for n2 in n2_grid:
            if n1 >= n2:
                continue

            for alpha in alpha_grid:
                # 1. MACD Line 계산
                ema_short = generalized_ema(df["Close"], n1, alpha)
                ema_long = generalized_ema(df["Close"], n2, alpha)
                macd_line = ema_short - ema_long

                for n3 in n3_grid:
                    # 2. Signal Line 계산 (MACD Line의 EMA)
                    signal_line = generalized_ema(macd_line, n3, alpha)

                    # 3. 인덱스 맞춰서 CE 계산
                    common_idx = macd_line.dropna().index.intersection(
                        signal_line.dropna().index
                    )
                    if len(common_idx) == 0:
                        continue

                    ce_val = calculate_cross_entropy(
                        df["Close"].loc[common_idx],
                        macd_line.loc[common_idx],
                        signal_line.loc[common_idx],
                    )

                    if ce_val < best_ce:
                        best_ce = ce_val
                        best_params = {"n1": n1, "n2": n2, "n3": n3, "alpha": alpha}

    print("\n" + "=" * 40)
    print(f"[{ticker} 최적 MACD 파라미터]")
    print(f"1. 단기 윈도우 (n1): {best_params['n1']}")
    print(f"2. 장기 윈도우 (n2): {best_params['n2']}")
    print(f"3. 시그널 윈도우 (n3): {best_params['n3']}")
    print(f"4. 가중치 계수 (alpha): {best_params['alpha']}")
    print(f"5. 최소 크로스 엔트로피: {best_ce:.6f}")
    print("=" * 40)

    return best_params


if __name__ == "__main__":
    optimal_set = find_optimal_macd_full_params("AAPL", period="3y")
