
import numpy as np
from numpy.linalg import inv
import pandas as pd
import yfinance as yf

import warnings
warnings.filterwarnings("ignore", category=pd.errors.Pandas4Warning)

from models.markowitz import MarkowitzModel

# Example views
# AAPL 0.1 [up/down]
# GOOG 0.25 [over/under] NVDA

def is_asset(tok, prices):
    return tok in prices.columns

def is_valid_view(toks, prices):
    if not is_asset(toks[0], prices): return False 
    
    try: float(toks[1])
    except ValueError: return False
    
    if toks[2] not in ["up", "down", "over", "under"]:
        return False 
    elif toks[2] in ["up", "down"]:
        if len(toks) != 3 : return False
    else:
        if len(toks) != 4: return False
        if not is_asset(toks[3], prices): return False 

    return True

def parse_view(toks, prices):
    if len(toks) == 3:
        dir = 1 if toks[2] == "up" else -1
        picks = [dir if toks[0] == abbr else 0 for abbr in prices.columns]
        return float(toks[1]), picks
    else:
        abbrs = [toks[0], toks[3]]
        dirs = [1, -1] if toks[2] == "over" else [-1, 1]
        picks = []
        for abbr in prices.columns:
            if abbr == abbrs[0]: picks.append(dirs[0])
            elif abbr == abbrs[1]: picks.append(dirs[1])
            else: picks.append(0)
        return float(toks[1]), picks

def get_views(prices):
    Q = []; P = []
    while True:
        view = input("Enter a view or type \"done\". ").strip()
        if view.lower() == "done": break

        toks = view.split(" ")
        if not is_valid_view(toks, prices): 
            print("Invalid view statement. Use either")
            print("\"XYZ x.xx [up/down]\" or \"XYZ x.xx [over/under] ABC\".")
        else:
            weight, picks = parse_view(toks, prices)
            Q.append(weight)
            P.append(picks)
    
    print("")
    return np.array(Q).reshape(-1, 1), np.array(P)

def get_market_weights(prices):
    assets = list(prices.columns)
    market_caps = {}
    for asset in assets:
        info = yf.Ticker(asset).info
        market_cap = info.get("marketCap", None)
        if market_cap is None:
            shares = info.get("sharesOutstanding", None)
            price = info.get("regularMarketPrice", None) or info.get("previousClose", None)
            if shares is not None and price is not None:
                market_cap = shares * price
        if market_cap is None:
            market_cap = info.get("totalAssets", None)
        market_caps[asset] = market_cap
    
    market_caps = pd.Series(market_caps).values.astype(float)
    w = market_caps / sum(market_caps)
    return np.array(w).reshape(-1, 1)

class BlackLittermanModel(MarkowitzModel):
    def __init__(self, prices, short, penalty, penalty_weight):
        super().__init__(prices, short, penalty, penalty_weight)
        tau = 0.05
        Q, P = get_views(prices)
        Omega = np.diag(np.diag(tau * P @ self.Sigma @ P.T))
        
        w_mkt = get_market_weights(prices)
        
        r_m = self.returns.values @ w_mkt
        delta = r_m.mean() / r_m.var(ddof=1)
        Pi = delta * self.Sigma @ w_mkt

        self.mu = np.reshape(inv(inv(tau * self.Sigma) + P.T @ inv(Omega) @ P) @ (inv(tau * self.Sigma) @ Pi + P.T @ inv(Omega) @ Q), (self.num_stocks, 1))

    def print(self, portfolio_value):
        print(" -=-=-=- Black-Litterman Model -=-=-=- ")
        print(f"Maximum Sharpe Ratio: {self.max_sr:.4f}; Expected Return: {self.return_opt:.4f}; Expected Risk: {self.risk_opt:.4f}")
        print(f"Optimized Portfolio: \n {dict(zip(self.returns.columns, (self.omega_opt * portfolio_value).round(2).tolist()))}")