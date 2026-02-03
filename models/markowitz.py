# Requires solving a quadratic program
import numpy as np
from matplotlib import pyplot as plt
import warnings
import cvxpy as cp

class MarkowitzModel:
    def __init__(self, prices, short, confidence, penalty, penalty_weight):
        self.returns = prices.pct_change().dropna()
        num_stocks = len(prices.columns)
        mu = np.reshape(self.returns[-12:].mean(), (num_stocks, 1)) # use a shorter time window for mean returns
        Sigma = self.returns.cov()      

        omega_vec = []; self.objective_values = []

        if confidence != "none":
            print(f"Unused parameter: {confidence=}")
        if penalty == "none" and penalty_weight not in [0, "none"]:
            print(f"Unused parameter: {penalty_weight=}")

        self.r_range = np.linspace(mu.min()+1e-3, mu.max()-1e-3, 500).tolist()

        omega = cp.Variable(num_stocks)
        f = cp.quad_form(omega, Sigma) + penalty_weight * penalty(omega)

        def g(r, omega, mu):
            constraints = [mu.T @ omega == r, sum(omega) == 1]
            if not short: constraints.append(omega >= 0)
            return constraints

        invalid_r = []
        for r in self.r_range:
            prob = cp.Problem(cp.Minimize(f), g(r, omega, mu))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                prob.solve()
            if prob.status == "optimal":
                omega_vec.append(omega.value)
                self.objective_values.append(omega.value @ (Sigma @ omega.value))
            else:
                #print(f"Problem {prob.status} for {r=}")
                invalid_r.append(r)
        for r in invalid_r:
            self.r_range.remove(r)

        sr = np.array(self.r_range) / self.objective_values
        idx = sr.argmax()
        self.max_sr = sr[idx]
        self.omega_opt = omega_vec[idx]
        self.return_opt = self.r_range[idx]
        self.risk_opt = self.objective_values[idx]

    def print(self, portfolio_value):
        print(" -=-=-=- Mean-Variance Model -=-=-=- ")
        print(f"Maximum Sharpe Ratio: {self.max_sr:.4f}; Expected Return: {self.return_opt:.4f}; Expected Risk: {self.risk_opt:.4f}")
        print(f"Optimized Portfolio: \n {dict(zip(self.returns.columns, (self.omega_opt * portfolio_value).round(2).tolist()))}")

    def plot(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.objective_values, self.r_range)
        plt.grid(True)
        plt.ylabel("Expected Return")
        plt.xlabel("Expected Risk")
        risk_values = np.linspace(0, self.risk_opt, 100)
        plt.plot(risk_values, risk_values * self.max_sr, color="black")
        plt.title("Mean-Variance Efficient Frontier")
        plt.show()

def markowitz(prices, portfolio_value, short, confidence, penalty, penalty_weight, plot):
    model = MarkowitzModel(prices, short, confidence, penalty, penalty_weight)
    model.print(portfolio_value)
    if plot: model.plot()
