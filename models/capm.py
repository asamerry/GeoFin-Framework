import numpy as np
from models.markowitz import MarkowitzModel

class CAPModel(MarkowitzModel):
    def __init__(self, prices, portfolio_value, short, penalty, penalty_weight, rf, views_file, recache):
        super().__init__(prices, portfolio_value, short, penalty, penalty_weight, rf, "none", False)
        self.title = "Capital Asset Pricing"
        beta = np.array(self.Sigma.mean() / self.Sigma.values.sum())
        rm = self.mu.mean()
        self.mu = (rf + beta * (rm - rf)).reshape(-1, 1)