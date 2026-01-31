import os
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# =========================
# Style 
# =========================
def set_dark_plot_style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "axes.facecolor": "#000000",
        "figure.facecolor": "#000000",
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "grid.color": "#333333",
        "legend.facecolor": "#111111",
        "legend.edgecolor": "#222222",
        "font.size": 10,
    })


# =========================
# Generalized EMA 
# =========================
def generalized_ema_fast(series, n, alpha):
    if len(series) < n:
        return pd.Series([np.nan] * len(series), index=series.index)
    weights = np.array([alpha**i for i in range(n)])[::-1]
    weights /= weights.sum()
    res = np.convolve(series.values, weights, mode="valid")
    full_res = np.empty(len(series))
    full_res[:n-1] = np.nan
    full_res[n-1:] = res
    return pd.Series(full_res, index=series.index)


# =========================
# Typical EMA for (12,26,9)
# =========================
def typical_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


# =========================
# Build MACD lines
# =========================
def build_generalized_macd(close, n1, n2, n3, alpha):
    ema_s = generalized_ema_fast(close, n1, alpha)
    ema_l = generalized_ema_fast(close, n2, alpha)
    macd = ema_s - ema_l
    sig = generalized_ema_fast(macd.dropna(), n3, alpha).reindex(macd.index)
    return macd, sig


def build_typical_macd(close, n1=12, n2=26, n3=9):
    ema_s = typical_ema(close, n1)
    ema_l = typical_ema(close, n2)
    macd = ema_s - ema_l
    sig = typical_ema(macd, n3)
    return macd, sig


# =========================
# Signals + Portfolio 
# =========================
def make_signals_and_portfolio(df_plot: pd.DataFrame) -> pd.DataFrame:
    out = df_plot.copy()
    out["diff"] = out["MACD"] - out["Signal"]
    out["diff_prev"] = out["diff"].shift(1)

    out["Buy_Signal"] = ((out["diff_prev"] <= 0) & (out["diff"] > 0)).astype(int)
    out["Sell_Signal"] = ((out["diff_prev"] >= 0) & (out["diff"] < 0)).astype(int)

    pos = np.zeros(len(out), dtype=float)
    in_pos = False
    for i in range(len(out)):
        if (not in_pos) and out["Buy_Signal"].iloc[i] == 1:
            in_pos = True
        elif in_pos and out["Sell_Signal"].iloc[i] == 1:
            in_pos = False
        pos[i] = 1.0 if in_pos else 0.0
    out["Position"] = pos

    out["ret"] = out["Close"].pct_change().fillna(0.0)
    out["strategy_ret"] = out["Position"].shift(1).fillna(0.0) * out["ret"]
    out["Portfolio"] = (1.0 + out["strategy_ret"]).cumprod()

    out["Hist"] = out["diff"]
    return out


def compute_key_metrics(df_strat: pd.DataFrame, close_full: pd.Series, annualization: int = 252) -> dict:
    """
    df_strat: make_signals_and_portfolio()를 거친 df (date, Close, strategy_ret, Portfolio, Buy_Signal, Sell_Signal 포함)
    close_full: 원본 close (Buy&Hold 계산용)
    """
    m = {}

    # 전략 누적
    equity = df_strat["Portfolio"].astype(float)
    m["final_value"] = float(equity.iloc[-1])
    m["total_return_pct"] = float((equity.iloc[-1] - 1.0) * 100.0)

    # 일간 수익률
    r = df_strat["strategy_ret"].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    n = len(r)
    years = max(n / annualization, 1e-12)

    # CAGR(근사)
    m["cagr_pct"] = float((equity.iloc[-1] ** (1.0 / years) - 1.0) * 100.0)

    # Vol / Sharpe (rf=0 가정)
    vol = r.std(ddof=1) * np.sqrt(annualization) if n > 1 else np.nan
    m["vol_pct"] = float(vol * 100.0) if np.isfinite(vol) else np.nan
    mean_ann = r.mean() * annualization if n > 0 else np.nan
    sharpe = mean_ann / vol if (np.isfinite(vol) and vol > 0) else np.nan
    m["sharpe"] = float(sharpe) if np.isfinite(sharpe) else np.nan

    # Max Drawdown
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    m["max_drawdown_pct"] = float(dd.min() * 100.0)

    # 트레이드 수 (신호 개수)
    m["buy_signals"] = int(df_strat["Buy_Signal"].sum())
    m["sell_signals"] = int(df_strat["Sell_Signal"].sum())

    # (간단) 체결 트레이드 수/승률: buy 이후 첫 sell로 페어링
    buys = df_strat.index[df_strat["Buy_Signal"] == 1].to_list()
    sells = df_strat.index[df_strat["Sell_Signal"] == 1].to_list()
    trade_rets = []
    i = j = 0
    while i < len(buys) and j < len(sells):
        b = buys[i]
        while j < len(sells) and sells[j] <= b:
            j += 1
        if j >= len(sells):
            break
        s = sells[j]
        entry = float(df_strat.loc[b, "Close"])
        exit_ = float(df_strat.loc[s, "Close"])
        trade_rets.append(exit_ / entry - 1.0)
        i += 1
        j += 1

    m["closed_trades"] = int(len(trade_rets))
    if len(trade_rets) > 0:
        tr = np.array(trade_rets, dtype=float)
        m["win_rate_pct"] = float((tr > 0).mean() * 100.0)
        m["avg_trade_ret_pct"] = float(tr.mean() * 100.0)
    else:
        m["win_rate_pct"] = np.nan
        m["avg_trade_ret_pct"] = np.nan

    # Buy & Hold
    bh = (1.0 + close_full.pct_change().fillna(0.0)).cumprod()
    m["bh_final"] = float(bh.iloc[-1])
    m["bh_return_pct"] = float((bh.iloc[-1] - 1.0) * 100.0)

    return m


def print_key_report(ticker: str, best: dict, m_opt: dict, m_base: dict):
    print("\n" + "=" * 60)
    print(f"[{ticker} | Key Metrics]")
    print(f"Params: n1={best['n1']} n2={best['n2']} n3={best['n3']} alpha={best['alpha']}")
    print("-" * 60)

    def line(tag, m):
        return (
            f"{tag:<18} final={m['final_value']:.4f}  "
            f"tot={m['total_return_pct']:.2f}%  "
            f"CAGR~{m['cagr_pct']:.2f}%  "
            f"Vol~{m['vol_pct']:.2f}%  "
            f"Sharpe~{m['sharpe']:.3f}  "
            f"MDD={m['max_drawdown_pct']:.2f}%"
        )

    print(line("Optimized", m_opt))
    print(line("Typical(12/26/9)", m_base))
    print(f"{'Buy&Hold':<18} final={m_opt['bh_final']:.4f}  ret={m_opt['bh_return_pct']:.2f}%")
    print("-" * 60)

    print(f"Optimized trades: closed={m_opt['closed_trades']}  "
          f"buy_sig={m_opt['buy_signals']} sell_sig={m_opt['sell_signals']}  "
          f"win={m_opt['win_rate_pct']:.2f}%  avg_trade={m_opt['avg_trade_ret_pct']:.2f}%")
    print(f"Typical   trades: closed={m_base['closed_trades']}  "
          f"buy_sig={m_base['buy_signals']} sell_sig={m_base['sell_signals']}  "
          f"win={m_base['win_rate_pct']:.2f}%  avg_trade={m_base['avg_trade_ret_pct']:.2f}%")
    print("=" * 60)



# =========================
# Figure1: price/macd/hist 
# =========================
def plot_macd_bundle_single(df_plot: pd.DataFrame, title: str = "", save_dir: str | None = "figures",
                           file_prefix: str = "opt"):
    set_dark_plot_style()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1]},
        figsize=(16, 9)
    )

    ax1.plot(df_plot["date"], df_plot["Close"], linewidth=1.6, label="Close")
    buys = df_plot[df_plot["Buy_Signal"] == 1]
    sells = df_plot[df_plot["Sell_Signal"] == 1]
    ax1.scatter(buys["date"], buys["Close"], marker="^", s=90, label="Buy", zorder=5)
    ax1.scatter(sells["date"], sells["Close"], marker="v", s=90, label="Sell", zorder=5)
    ax1.set_ylabel("Price")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(loc="upper left")
    if title:
        ax1.set_title(title, fontsize=14)

    ax2.plot(df_plot["date"], df_plot["MACD"], linewidth=1.2, label="MACD")
    ax2.plot(df_plot["date"], df_plot["Signal"], linewidth=1.2, label="Signal")
    ax2.axhline(0, linestyle="--", linewidth=1)
    ax2.set_ylabel("MACD")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(loc="upper left")

    colors = np.where(df_plot["Hist"] >= 0, "#00FF88", "#FF5555")
    ax3.bar(df_plot["date"], df_plot["Hist"], color=colors, width=1)
    ax3.axhline(0, linestyle="--", linewidth=1)
    ax3.set_ylabel("Hist")
    ax3.grid(True, linestyle="--", alpha=0.3)

    ax3.xaxis.set_major_locator(mdates.YearLocator())
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_dir is not None:
        f1 = os.path.join(save_dir, f"{file_prefix}_figure1_price_macd_hist.png")
        fig.savefig(f1, dpi=200, bbox_inches="tight")

    plt.show()

    if save_dir is not None:
        print(f"Saved: {f1}")


# =========================
# Figure2: Portfolio 비교 (여기서 두 세트 비교)
# =========================
def plot_portfolio_compare(df_opt: pd.DataFrame, df_base: pd.DataFrame, close_series: pd.Series,
                           title: str = "", save_dir: str | None = "figures"):
    set_dark_plot_style()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # buy&hold
    bh = (1.0 + close_series.pct_change().fillna(0.0)).cumprod()

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(df_opt["date"], df_opt["Portfolio"], linewidth=2.2, label="Portfolio (Optimized)")
    ax.plot(df_base["date"], df_base["Portfolio"], linewidth=2.2, linestyle="--", label="Portfolio (12,26,9)")
    ax.plot(bh.index, bh.values, linewidth=1.6, linestyle=":", label="Buy & Hold")

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title("Portfolio Value Comparison (next-bar execution)", fontsize=14)

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_dir is not None:
        f2 = os.path.join(save_dir, "figure2_portfolio_compare.png")
        fig.savefig(f2, dpi=200, bbox_inches="tight")
        print(f"Saved: {f2}")

    plt.show()


# =========================
# (선택) Figure3: Price에 최소 마커로 두 전략 비교
# =========================
def plot_price_signals_compare(df_opt: pd.DataFrame, df_base: pd.DataFrame, title: str = "",
                               save_dir: str | None = "figures"):
    set_dark_plot_style()
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(df_opt["date"], df_opt["Close"], linewidth=1.6, label="Close")

    opt_buys = df_opt[df_opt["Buy_Signal"] == 1]
    opt_sells = df_opt[df_opt["Sell_Signal"] == 1]
    base_buys = df_base[df_base["Buy_Signal"] == 1]
    base_sells = df_base[df_base["Sell_Signal"] == 1]

    # Optimized: 큰 삼각형
    ax.scatter(opt_buys["date"], opt_buys["Close"], marker="^", s=90, label="Buy (Optimized)", zorder=5)
    ax.scatter(opt_sells["date"], opt_sells["Close"], marker="v", s=90, label="Sell (Optimized)", zorder=5)

    # Typical: 작은 점/엑스 (겹침 최소화)
    ax.scatter(base_buys["date"], base_buys["Close"], marker="o", s=28, label="Buy (12,26,9)", zorder=6)
    ax.scatter(base_sells["date"], base_sells["Close"], marker="x", s=35, label="Sell (12,26,9)", zorder=6)

    ax.set_ylabel("Price")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
    ax.set_title(title if title else "Price + Signals (Optimized vs 12,26,9)", fontsize=14)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()
    plt.tight_layout()

    if save_dir is not None:
        f3 = os.path.join(save_dir, "figure3_price_signals_compare.png")
        fig.savefig(f3, dpi=200, bbox_inches="tight")
        print(f"Saved: {f3}")

    plt.show()


# =========================
# Example run
# =========================
def compute_cumret_crossover(close: pd.Series, macd: pd.Series, sig: pd.Series) -> float:
    """
    make_signals_and_portfolio()와 동일한 로직(크로스오버 토글 + next-bar execution)로
    최종 누적수익(cumprod 마지막 값)을 반환.
    """
    df = pd.DataFrame({"Close": close, "MACD": macd, "Signal": sig}).dropna()
    if len(df) < 5:
        return -np.inf  # 데이터가 너무 짧으면 탈락

    diff = df["MACD"] - df["Signal"]
    diff_prev = diff.shift(1)

    buy = (diff_prev <= 0) & (diff > 0)
    sell = (diff_prev >= 0) & (diff < 0)

    pos = np.zeros(len(df), dtype=float)
    in_pos = False
    for i in range(len(df)):
        if (not in_pos) and buy.iloc[i]:
            in_pos = True
        elif in_pos and sell.iloc[i]:
            in_pos = False
        pos[i] = 1.0 if in_pos else 0.0

    ret = df["Close"].pct_change().fillna(0.0)
    strat_ret = pd.Series(pos, index=df.index).shift(1).fillna(0.0) * ret
    return float((1.0 + strat_ret).cumprod().iloc[-1])


def find_best_params_by_return(ticker, period="2y"):
    df = yf.download(ticker, period=period, multi_level_index=False)
    close = df["Close"].astype(float)
    daily_ret = close.pct_change()

    n1_range = list(range(1, 22))
    n2_range = list(range(21, 31))
    n3_range = list(range(1, 18))
    alpha_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    best_return = -np.inf
    best_params = {}

    for n1 in n1_range:
        for n2 in n2_range:
            if n1 >= n2:
                continue
            for alpha in alpha_range:
                ema_s = generalized_ema_fast(close, n1, alpha)
                ema_l = generalized_ema_fast(close, n2, alpha)
                macd = ema_s - ema_l
                for n3 in n3_range:
                    sig = generalized_ema_fast(macd.dropna(), n3, alpha).reindex(macd.index)
                    #pos = (macd > sig).astype(int)
                    #strat_ret = pos.shift(1).fillna(0) * daily_ret
                    #cum_ret = (1 + strat_ret).cumprod().iloc[-1]
                    cum_ret = compute_cumret_crossover(close, macd, sig)


                    if cum_ret > best_return:
                        best_return = cum_ret
                        best_params = {"n1": n1, "n2": n2, "n3": n3, "alpha": alpha}

    return best_params


if __name__ == "__main__":
    ticker = "005935.KS"
    period = "2y"

    best = find_best_params_by_return(ticker, period=period)

    df = yf.download(ticker, period=period, multi_level_index=False)
    close = df["Close"].astype(float)

    # Optimized lines
    macd_opt, sig_opt = build_generalized_macd(close, best["n1"], best["n2"], best["n3"], best["alpha"])

    df_opt = pd.DataFrame({
        "date": close.index,
        "Close": close.values,
        "MACD": macd_opt.values,
        "Signal": sig_opt.values,
    }).dropna().reset_index(drop=True)
    df_opt = make_signals_and_portfolio(df_opt)

    # Typical(12,26,9) lines
    macd_base, sig_base = build_typical_macd(close, 12, 26, 9)

    df_base = pd.DataFrame({
        "date": close.index,
        "Close": close.values,
        "MACD": macd_base.values,
        "Signal": sig_base.values,
    }).dropna().reset_index(drop=True)
    df_base = make_signals_and_portfolio(df_base)

        # ---- Key Metrics (TEXT) ----
    m_opt = compute_key_metrics(df_opt, close)
    m_base = compute_key_metrics(df_base, close)
    print_key_report(ticker, best, m_opt, m_base)






    # ---- Figures ----
    title1 = (f"{ticker} | Optimized(mutant): n1={best['n1']} n2={best['n2']} "
              f"n3={best['n3']} alpha={best['alpha']}")

    plot_macd_bundle_single(df_opt, title=title1, save_dir="figures", file_prefix="opt")

    plot_portfolio_compare(
        df_opt, df_base, close,
        title=f"{ticker} | Portfolio: Optimized vs Typical(12,26,9) vs Buy&Hold",
        save_dir="figures"
    )

    # 필요하면 켜기(겹침이 덜한 신호 비교)
    plot_price_signals_compare(
        df_opt, df_base,
        title=f"{ticker} | Price + Signals: Optimized vs Typical(12,26,9)",
        save_dir="figures"
    )
