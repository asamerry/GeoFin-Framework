class CVaRModel:
    def __init__(self):
        self.alpha = 0.95 # 0.90, 0.95, 0.99


def cvar(prices, num_stocks, short, portfolio_value, plot):
    # VaR = argmin_v{P(L <= v) >= alpha}
    # CVaR = E[L | L >= VaR]
    returns = prices.pct_change().dropna()

    model = CVaRModel()
    model.print()
    if plot: model.plot()