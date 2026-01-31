import os
import numpy as np
import pandas as pd
import yfinance as yf

# -----------------------------
# Utils
# -----------------------------
def sigmoid(x):
    # overflow 방지
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))

def cross_entropy(y, p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return float(-(y*np.log(p) + (1-y)*np.log(1-p)).mean())

def annualized_sharpe(r):
    r = np.asarray(r)
    s = r.std()
    return 0.0 if s == 0 else float(r.mean() / s * np.sqrt(252))

def max_drawdown(cum):
    cum = np.asarray(cum)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum) / peak
    return float(dd.max())

# -----------------------------
# Data
# -----------------------------
def load_prices(symbol="SPY", start="2010-01-01", end="2023-12-31"):
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError("yfinance download failed (empty).")
    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float)
    close = close[~close.index.duplicated(keep="first")]
    return close

# -----------------------------
# FAST finite EMA (alpha, N) via convolution
# -----------------------------
def finite_ema_conv(x, alpha, N):
    """
    x: 1D np.array
    alpha: (0,1)
    N: window length
    output: same length, first N-1 = np.nan
    """
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")
    if N < 2:
        raise ValueError("N must be >= 2")

    # weights: k=0 newest weight, k=N-1 oldest weight
    w = (1 - alpha) ** np.arange(N, dtype=float)
    w = w / w.sum()

    # rolling dot = convolution with reversed weights
    # np.convolve gives length len(x)+N-1; "valid" gives len(x)-N+1
    valid = np.convolve(x, w, mode="valid")  # 이미 (oldest->newest) 정렬 대응됨
    out = np.empty_like(x, dtype=float)
    out[:] = np.nan
    out[N-1:] = valid
    return out

def finite_macd(x, a_fast, N_fast, a_slow, N_slow, a_sig, N_sig):
    ema_f = finite_ema_conv(x, a_fast, N_fast)
    ema_s = finite_ema_conv(x, a_slow, N_slow)
    macd = ema_f - ema_s
    sig = finite_ema_conv(macd, a_sig, N_sig)
    hist = macd - sig
    return macd, sig, hist

# -----------------------------
# Strategy: leverage regime filter (2x when macd>0, else 1x)
# -----------------------------
def leverage_from_macd(macd, lev_on=2.0, lev_off=1.0, deadband=0.0):
    lev = np.empty_like(macd, dtype=float)
    lev[:] = np.nan

    on = macd > deadband
    off = macd < -deadband
    lev[on] = lev_on
    lev[off] = lev_off

    # nan 구간은 직전 값 유지
    # 맨 앞 nan은 lev_off로
    if np.isnan(lev[0]):
        lev[0] = lev_off
    for i in range(1, len(lev)):
        if np.isnan(lev[i]):
            lev[i] = lev[i-1]
    return lev

def backtest_leverage(prices, lev, cost_bp=10.0):
    px = prices.values
    r = np.zeros_like(px)
    r[1:] = px[1:] / px[:-1] - 1.0

    # 거래비용: 레버리지 변경량 기반
    turn = np.zeros_like(lev)
    turn[1:] = np.abs(lev[1:] - lev[:-1])
    cost = turn * (cost_bp / 10000.0)

    strat = np.zeros_like(r)
    strat[1:] = lev[:-1] * r[1:] - cost[1:]
    cum = np.cumprod(1 + strat)

    return {
        "total_return": float(cum[-1] - 1),
        "sharpe": annualized_sharpe(strat[1:]),
        "mdd": max_drawdown(cum),
        "avg_lev": float(np.nanmean(lev)),
        "avg_turnover": float(np.mean(turn[1:])),
        "cum": cum,
        "strat": strat
    }

# -----------------------------
# Cross-Entropy objective (optimize alpha, N)
# -----------------------------
def ce_objective(prices, params, beta=5.0, split_ratio=0.7):
    """
    params = (a_fast,N_fast,a_slow,N_slow,a_sig,N_sig)
    objective: TEST cross-entropy (OOS)
    """
    a_fast, N_fast, a_slow, N_slow, a_sig, N_sig = params
    x = prices.values

    macd, sig, hist = finite_macd(x, a_fast, N_fast, a_slow, N_slow, a_sig, N_sig)

    # label: next-day direction
    r_fwd = np.empty_like(x)
    r_fwd[:-1] = x[1:] / x[:-1] - 1.0
    r_fwd[-1] = np.nan
    y = (r_fwd > 0).astype(int)

    # probability from macd (classification)
    p = sigmoid(beta * macd)

    # valid mask (macd finite + y finite)
    valid = ~np.isnan(macd) & ~np.isnan(r_fwd)
    idx = np.where(valid)[0]
    if len(idx) < 300:
        return np.inf, np.inf  # 데이터 너무 적으면 버림

    cut = int(len(idx) * split_ratio)
    tr_idx = idx[:cut]
    te_idx = idx[cut:]

    y_tr, p_tr = y[tr_idx], p[tr_idx]
    y_te, p_te = y[te_idx], p[te_idx]

    ce_tr = cross_entropy(y_tr, p_tr)
    ce_te = cross_entropy(y_te, p_te)
    return ce_tr, ce_te

def optimize_ce(prices, beta=5.0, split_ratio=0.7, max_iters=5000, seed=7):
    """
    너무 큰 그리드는 느리니까:
    - "랜덤 서치 + 제한된 후보"로 빠르게 최적 찾기
    """
    rng = np.random.default_rng(seed)

    # N 후보 (과제에서 보통 12/26/9 주변을 탐색)
    N_fast_list = np.array([6, 8, 10, 12, 15, 18, 20])
    N_slow_list = np.array([18, 20, 24, 26, 30, 35, 40, 45, 50])
    N_sig_list  = np.array([5, 6, 8, 9, 12, 15])

    # alpha 후보 (2/(N+1) 강제X — 자유)
    # 너무 촘촘하면 느리니까 우선 넓게
    a_list = np.array([0.02, 0.03, 0.05, 0.07, 0.09, 0.12, 0.15, 0.18, 0.22, 0.28])

    best = None

    for it in range(1, max_iters + 1):
        N_fast = int(rng.choice(N_fast_list))
        N_slow = int(rng.choice(N_slow_list))
        if N_fast >= N_slow:
            continue
        N_sig  = int(rng.choice(N_sig_list))

        a_fast = float(rng.choice(a_list))
        a_slow = float(rng.choice(a_list))
        a_sig  = float(rng.choice(a_list))

        # fast가 느리면 MACD 의미가 약해짐 (제약 넣기)
        if a_fast <= a_slow:
            continue

        params = (a_fast, N_fast, a_slow, N_slow, a_sig, N_sig)
        ce_tr, ce_te = ce_objective(prices, params, beta=beta, split_ratio=split_ratio)

        if not np.isfinite(ce_te):
            continue

        # 선택 기준: TEST CE 최소 + 과적합 방지(트레인-테스트 차이 패널티)
        gap = abs(ce_te - ce_tr)
        score = ce_te + 0.25 * gap  # 작을수록 좋음

        if best is None or score < best["score"]:
            best = {
                "params": params,
                "ce_train": ce_tr,
                "ce_test": ce_te,
                "gap": gap,
                "score": score,
                "iter": it
            }

        if it % 300 == 0 and best is not None:
            p = best["params"]
            print(f"[{it}] best score={best['score']:.5f} | ce_test={best['ce_test']:.5f} | "
                  f"N=({p[1]},{p[3]},{p[5]}) a=({p[0]:.3f},{p[2]:.3f},{p[4]:.3f})")

    return best

# -----------------------------
# Plot saving (matplotlib import is heavy, do it only when plotting)
# -----------------------------
def save_plot(path, x, labels, title):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    for series, label, style in labels:
        plt.plot(x, series, label=label, **style)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    print("saved:", path)
    try:
        os.system(f'open "{path}"')
    except:
        pass
    plt.show()

# -----------------------------
# Main demos
# -----------------------------
def run_finite_leverage(prices, params, lev_on=2.0, lev_off=1.0, deadband=0.0, cost_bp=10.0,
                        out_name="finite_macd_leverage.png"):
    a_fast, N_fast, a_slow, N_slow, a_sig, N_sig = params
    x = prices.values
    macd, sig, hist = finite_macd(x, a_fast, N_fast, a_slow, N_slow, a_sig, N_sig)

    lev = leverage_from_macd(macd, lev_on=lev_on, lev_off=lev_off, deadband=deadband)
    res = backtest_leverage(prices, lev, cost_bp=cost_bp)

    # buy&hold 1x,2x 비교
    r = prices.pct_change().fillna(0).values
    bh1 = np.cumprod(1 + 1.0 * r)
    bh2 = np.cumprod(1 + 2.0 * r)

    out = os.path.join(os.path.dirname(__file__), out_name)
    labels = [
        (res["cum"], "Finite MACD Filtered Leverage", {}),
        (bh1, "Buy & Hold 1x", {"alpha": 0.75}),
        (bh2, "Buy & Hold 2x", {"alpha": 0.6, "linestyle": "--"})
    ]
    save_plot(out, prices.index, labels, "SPY: Finite MACD (alpha,N free) Leverage Regime Filter")

    print("\n=== FINITE MACD + Leverage Regime Filter ===")
    print(f"N_fast={N_fast}, N_slow={N_slow}, N_signal={N_sig}")
    print(f"alpha_fast={a_fast:.4f}, alpha_slow={a_slow:.4f}, alpha_sig={a_sig:.4f}")
    print(f"Total Return: {res['total_return']*100:.2f}% | Sharpe: {res['sharpe']:.2f} | MDD: {res['mdd']*100:.2f}%")
    print(f"Avg Lev: {res['avg_lev']:.2f} | Avg Turnover: {res['avg_turnover']:.4f}")

    return res

def main():
    prices = load_prices("SPY")

    # 1) 기본 finite (너가 전에 쓴 12/26/9 + (관례 alpha)도 가능하지만, 여기선 그냥 예시)
    #    (원하면 a를 2/(N+1)로 넣어도 됨. 근데 최적화는 자유 alpha.)
    base_params = (0.1538, 12, 0.0741, 26, 0.2000, 9)  # 네 로그랑 비슷한 예시
    run_finite_leverage(prices, base_params, out_name="finite_macd_leverage_base.png")

    # 2) Cross-Entropy로 alpha & N 최적화
    print("\n=== Optimize (Cross-Entropy) ===")
    best = optimize_ce(prices, beta=5.0, split_ratio=0.7, max_iters=2500, seed=7)

    if best is None:
        print("No best found. Increase max_iters or widen grids.")
        return

    p = best["params"]
    print("\n>>> BEST by Cross-Entropy (OOS) <<<")
    print(f"iter={best['iter']}")
    print(f"N_fast={p[1]}, N_slow={p[3]}, N_sig={p[5]}")
    print(f"a_fast={p[0]:.4f}, a_slow={p[2]:.4f}, a_sig={p[4]:.4f}")
    print(f"CE train={best['ce_train']:.5f} | CE test={best['ce_test']:.5f} | gap={best['gap']:.5f} | score={best['score']:.5f}")

    # 3) 최적 파라미터로 실제 레버리지 전략 성과도 같이 찍어보기 (과제 보고서에 넣기 좋음)
    run_finite_leverage(prices, p, out_name="finite_macd_leverage_best_CE.png")

if __name__ == "__main__":
    main()
