import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import warnings

from utils import get_returns, get_risk

PENALTIES = {
    "none": lambda x: 0, 
    "l1": cp.norm1, 
    "l2": cp.sum_squares,
}

class _Optimizer:
    def __init__(self, prices, portfolio_value, return_est, risk_est, short, penalty, penalty_weight, rf, views_file, recache):
        self.prices = prices
        returns_historic = prices.pct_change().dropna()
        self.portfolio_value = portfolio_value
        self.return_est = return_est
        self.risk_est = risk_est
        self.short = short
        
        self.penalty = penalty
        self.penalty_func = PENALTIES[penalty]
        if penalty == "none" and penalty_weight != 0:
            if penalty_weight != "none":
                print(f"Unused parameter: {penalty_weight=}")
            self.penalty_weight = 0
        else: self.penalty_weight = penalty_weight

        self.rf = rf
        self.views_file = views_file
        self.recache = recache

        self.expected_risk = get_risk(risk_est, returns_historic)
        self.expected_returns = get_returns(return_est, returns_historic, self.expected_risk, rf, views_file, recache)

class MarkowitzOptimizer(_Optimizer):
    def __init__(self, prices, portfolio_value, return_est, risk_est, short, penalty, penalty_weight, rf, views_file, recache):
        super().__init__(prices, portfolio_value, return_est, risk_est, short, penalty, penalty_weight, rf, views_file, recache)
        self.omega_vec = []; self.objective_values = []
        self.omega = cp.Variable(len(self.prices.columns))
        self.f = cp.quad_form(self.omega, self.expected_risk) + self.penalty_weight * self.penalty_func(self.omega)

    def solve(self):
        print("Solving optimization problem ...")
        def g(r, omega, returns):
            constraints = [returns.T @ omega >= r, sum(omega) == 1]
            if not self.short: constraints.append(omega >= 0)
            return constraints

        self.r_range = np.linspace(max(self.expected_returns.min() + 1e-5, 0), self.expected_returns.max() - 1e-5, 500).tolist()
        invalid_r = []
        for r in self.r_range:
            prob = cp.Problem(cp.Minimize(self.f), g(r, self.omega, self.expected_returns))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                prob.solve()
            if prob.status == "optimal":
                self.omega_vec.append(self.omega.value)
                self.objective_values.append(self.omega.value @ self.expected_risk @ self.omega.value)
            else:
                # print(f"Problem {prob.status} for {r=}")
                invalid_r.append(r)
        for r in invalid_r:
            self.r_range.remove(r)

        sr = np.array(self.r_range) / self.objective_values
        idx = sr.argmax()
        self.max_sr = sr[idx]
        self.omega_opt = self.omega_vec[idx]
        self.return_opt = self.r_range[idx]
        self.risk_opt = self.objective_values[idx]
        self.portfolio = dict(zip(self.prices.columns, (self.omega_opt * self.portfolio_value).round(2).tolist()))
        self.portfolio = str(self.portfolio)[1:-1].replace("\'", "")

    def print(self):
        print(f"\n -=-=-=- Markowitz Model -=-=-=- ")
        print(f"Returns: {self.return_est.capitalize()}; Risk: {self.risk_est.capitalize()}")
        if self.penalty != "none":
            print(f"Penalty: {self.penalty}; Penalty Weight: {self.penalty_weight}")
        print(f"Maximum Sharpe Ratio: {self.max_sr:.4f}; Expected Return: {self.return_opt:.4f}; Expected Risk: {self.risk_opt:.4f}")
        print(f"Optimized Portfolio: \n{self.portfolio}")

    def plot(self, save, save_file="exports/out.png"):
        plt.figure(figsize=(12, 6))
        plt.plot(self.objective_values, self.r_range)
        plt.grid(True)
        plt.ylabel("Expected Return")
        plt.xlabel("Expected Risk")
        risk_values = np.linspace(0, self.risk_opt, 100)
        plt.plot(risk_values, risk_values * self.max_sr, color="black")
        plt.title(f"Markowitz Efficient Frontier")
        if save: plt.savefig(save_file.replace("txt", "png"))
        else: plt.show()