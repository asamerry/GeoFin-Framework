# Requires solving a quadratic program
import numpy as np
from matplotlib import pyplot as plt
import warnings
import cvxpy as cp

class MarkowitzModel:
    def __init__(self, prices, portfolio_value, short, penalty, penalty_weight, views_file, recache=False):
        self.portfolio_value = portfolio_value
        self.short = short
        self.title = "Markowitz"
        self.returns = prices.pct_change().dropna()
        self.num_stocks = len(prices.columns)
        self.mu = np.reshape(self.returns[-12:].mean(), (self.num_stocks, 1)) # use a shorter time window for mean returns
        self.Sigma = self.returns.cov()      

        self.omega_vec = []; self.objective_values = []

        self.r_range = np.linspace(self.mu.min()+1e-3, self.mu.max()-1e-3, 500).tolist()

        if views_file != "none":
            print(f"Unused parameter: {views_file=}")
        if penalty == "none" and penalty_weight not in [0, "none"]:
            print(f"Unused parameter: {penalty_weight=}")

        self.omega = cp.Variable(self.num_stocks)
        self.f = cp.quad_form(self.omega, self.Sigma) + penalty_weight * penalty(self.omega)

    def solve(self):
        print("Solving optimization problem ...")
        def g(r, omega, mu):
            constraints = [mu.T @ omega == r, sum(omega) == 1]
            if not self.short: constraints.append(omega >= 0)
            return constraints

        invalid_r = []
        for r in self.r_range:
            prob = cp.Problem(cp.Minimize(self.f), g(r, self.omega, self.mu))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                prob.solve()
            if prob.status == "optimal":
                self.omega_vec.append(self.omega.value)
                self.objective_values.append(self.omega.value @ self.Sigma @ self.omega.value)
            else:
                #print(f"Problem {prob.status} for {r=}")
                invalid_r.append(r)
        for r in invalid_r:
            self.r_range.remove(r)

        sr = np.array(self.r_range) / self.objective_values
        idx = sr.argmax()
        self.max_sr = sr[idx]
        self.omega_opt = self.omega_vec[idx]
        self.return_opt = self.r_range[idx]
        self.risk_opt = self.objective_values[idx]
        self.portfolio = dict(zip(self.returns.columns, (self.omega_opt * self.portfolio_value).round(2).tolist()))
        self.portfolio = str(self.portfolio)[1:-1].replace("\'", "")

    def print(self):
        print(f"\n -=-=-=- {self.title} Model -=-=-=- ")
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
        plt.title(f"{self.title} Efficient Frontier")
        if save: plt.savefig(save_file.replace("txt", "png"))
        else: plt.show()
