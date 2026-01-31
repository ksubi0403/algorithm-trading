import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# 1. 고속 EMA 함수 (기존 유지)
def generalized_ema_fast(series, n, alpha):
    if len(series) < n: return pd.Series([np.nan] * len(series), index=series.index)
    weights = np.array([alpha**i for i in range(n)])[::-1]
    weights /= weights.sum()
    res = np.convolve(series.values, weights, mode='valid')
    full_res = np.empty(len(series))
    full_res[:n-1] = np.nan
    full_res[n-1:] = res
    return pd.Series(full_res, index=series.index)

# 2. 파라미터 최적화 엔진 (수익률 기반)
def find_best_params_by_return(ticker, period="2y"):
    df = yf.download(ticker, period=period, multi_level_index=False)
    close = df['Close'].astype(float)
    daily_ret = close.pct_change()

    # 탐색 범위 (필요시 조정)
    n1_range = [1,2,3,4,5,6,7,8,9,10,11, 12,13, 14,15,16,17,18,19,20,21]
    n2_range = [21,22,23,24,25, 26,27,28,29, 30]
    n3_range = [1,2,3,4,5,6,7,8, 9,10,11,12, 13,14,15,16,17]
    alpha_range = [0.1, 0.2, 0.3, 0.4,0.5,0.6,0.7,0.8]

    best_return = -np.inf
    best_params = {}

    for n1 in n1_range:
        for n2 in n2_range:
            if n1 >= n2: continue
            for alpha in alpha_range:
                ema_s = generalized_ema_fast(close, n1, alpha)
                ema_l = generalized_ema_fast(close, n2, alpha)
                macd = ema_s - ema_l
                for n3 in n3_range:
                    sig = generalized_ema_fast(macd.dropna(), n3, alpha).reindex(macd.index)
                    pos = np.where(macd > sig, 1, 0)
                    strat_ret = pd.Series(pos, index=df.index).shift(1).fillna(0) * daily_ret
                    cum_ret = (1 + strat_ret).cumprod().iloc[-1]

                    if cum_ret > best_return:
                        best_return = cum_ret
                        best_params = {'n1': n1, 'n2': n2, 'n3': n3, 'alpha': alpha}
    return best_params

# 3. 시각화 함수 (파라미터 표시 강화)
def visualize_macd_with_returns(ticker, n1, n2, n3, alpha, period="2y"):
    df = yf.download(ticker, period=period, multi_level_index=False)
    close = df['Close'].astype(float)

    ema_s = generalized_ema_fast(close, n1, alpha)
    ema_l = generalized_ema_fast(close, n2, alpha)
    macd = ema_s - ema_l
    sig = generalized_ema_fast(macd.dropna(), n3, alpha).reindex(macd.index)
    hist = macd - sig

    df['Position'] = np.where(macd > sig, 1, 0)
    df['Strategy_Ret'] = df['Position'].shift(1).fillna(0) * close.pct_change()
    df['Cum_Strategy'] = (1 + df['Strategy_Ret']).cumprod()
    df['Cum_BuyHold'] = (1 + close.pct_change()).cumprod()

    final_strat = (df['Cum_Strategy'].iloc[-1] - 1) * 100
    final_bh = (df['Cum_BuyHold'].iloc[-1] - 1) * 100

    # 시각화
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 14), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1.5]})

    # 1. 주가 차트
    ax1.plot(df.index, close, color='gray', alpha=0.5)
    buy_marks = (df['Position'] == 1) & (df['Position'].shift(1) == 0)
    sell_marks = (df['Position'] == 0) & (df['Position'].shift(1) == 1)
    ax1.scatter(df.index[buy_marks], close[buy_marks], marker='^', color='#00ff00', s=120, label='BUY')
    ax1.scatter(df.index[sell_marks], close[sell_marks], marker='v', color='#ff0000', s=120, label='SELL')
    ax1.set_title(f"{ticker} Optimized Strategy", fontsize=16)

    # 파라미터 정보 텍스트 박스 추가
    param_text = (f"Optimized Parameters:\n"
                  f"n1 (Short): {n1}\n"
                  f"n2 (Long): {n2}\n"
                  f"n3 (Signal): {n3}\n"
                  f"Alpha: {alpha}\n\n"
                  f"Strategy Return: {final_strat:.2f}%\n"
                  f"Market Return: {final_bh:.2f}%")
    
    # 그래프 내부에 정보창 박스 생성
    ax1.text(0.02, 0.95, param_text, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7, edgecolor='white'))

    # 2. MACD 지표
    ax2.plot(df.index, macd, color='cyan', label='MACD')
    ax2.plot(df.index, sig, color='magenta', label='Signal')
    ax2.bar(df.index, hist, color=['#00ff00' if x > 0 else '#ff0000' for x in hist], alpha=0.3)
    ax2.legend(loc='upper left')

    # 3. 누적 수익률
    ax3.plot(df.index, df['Cum_Strategy'], color='#ffaa00', linewidth=2.5, label='Strategy')
    ax3.plot(df.index, df['Cum_BuyHold'], color='white', linestyle='--', alpha=0.5, label='Buy & Hold')
    ax3.set_title("Cumulative Returns Comparison")
    ax3.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

# 실행
if __name__ == "__main__":
    ticker = "005935.KS"
    best = find_best_params_by_return(ticker)
    visualize_macd_with_returns(ticker, **best)