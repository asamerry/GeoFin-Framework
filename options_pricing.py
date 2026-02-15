import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import norm
import io
from PIL import Image
from graphviz import Digraph

# Finite Difference Model
# GARCH Model
# Heston Model
# Jump Diffusion Model
# Local Volatility Model
# Monte Carlo Simulation
# Stochastic Volatility Model

class Option:
    def __init__(self, o_style, stock_p, strike_p, exp_t, sigma, int_r):
        self.o_style: str = o_style
        self.stock_p: float = stock_p
        self.strike_p: float = strike_p
        self.exp_t: float = exp_t
        self.sigma: float = sigma
        self.int_r: float = int_r

        self.call_p = 0
        self.put_p = 0

    def __repr__(self):
        repr = f"{self.o_style.capitalize()} Option"
        repr += f"\nInitial Price: {self.stock_p}; Strike Price: {self.strike_p}" 
        repr += f"\nExpiration Date: {self.exp_t}; Volatility: {self.sigma}; Interest Rate: {self.int_r}"
        repr += f"\nCall Price: {self.call_p:.2f}; Put Price: {self.put_p:.2f}"
        return repr

class _Pricer:
    def __init__(self):
        self.title = ""

    def _print(self, option):
        repr = f" -=-=-=- {self.title} Pricing Model -=-=-=- "
        repr += f"\n{option}"
        print(repr)
    
    def price(self, option, print: bool = True): pass # virtual
    
    def heatmap(self, options: list[list[Option]]):
        m = len(options); n = len(options[0])
        C = np.zeros((m, n)); P = np.zeros((m, n))
        for i in range(len(options)):
            for j in range(len(options[i])):
                self.price(options[i][j], print=False)
                C[i, j] = options[i][j].call_p
                P[i, j] = options[i][j].put_p

        x_vals = [options[0][j].stock_p for j in range(n)]
        y_vals = [options[i][0].sigma for i in range(m)]

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        c = ["darkred","red","lightcoral","white", "palegreen","green","darkgreen"]
        v = [0,.15,.4,.5,0.6,.9,1.]
        l = list(zip(v,c))
        cmap=LinearSegmentedColormap.from_list('rg',l, N=256)
        for ax, Z, title in zip(axes, [C, P], ["Call", "Put"]):
            im = ax.imshow(Z, origin="lower", cmap=cmap)
            
            ax.set_xticks(np.arange(n))
            ax.set_xticklabels([f"{x:.2f}" for x in x_vals])
            ax.set_yticks(np.arange(m))
            ax.set_yticklabels([f"{y:.2f}" for y in y_vals])

            ax.set_xlabel("Spot Price")
            ax.set_ylabel("Volatility")
            ax.set_title(title)

            fig.colorbar(im, ax=ax)

            for i in range(m):
                for j in range(n):
                    ax.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center", color="black")

        plt.tight_layout()
        plt.show()

class BinomialPricer(_Pricer):
    def __init__(self):
        self.title = "Binomial"

    def price(self, option: Option, print: bool = True, plot: bool = False, interval: float = 1):
        delta = 0
        u = np.exp((option.int_r - delta) * interval + option.sigma * np.sqrt(interval))
        d = np.exp((option.int_r - delta) * interval - option.sigma * np.sqrt(interval))
        p = (np.exp((option.int_r - delta) * interval) - d) / (u - d)

        S_arr = [np.array([option.stock_p])]
        for idx in range(option.exp_t):
            new = np.array([d * S_arr[idx][0]] + list(S_arr[idx] * u))
            S_arr.append(new)

        S_vals = S_arr[-1]
        C_vals = np.array([max(S - option.strike_p, 0) for S in S_vals])
        P_vals = np.array([max(option.strike_p - S, 0) for S in S_vals])

        C_arr = [C_vals]
        P_arr = [P_vals]

        while len(C_vals) > 1:
            C_vals = np.exp(-option.int_r * interval) * (C_vals[1:] * p + C_vals[:-1] * (1 - p))
            P_vals = np.exp(-option.int_r * interval) * (P_vals[1:] * p + P_vals[:-1] * (1 - p)) 

            C_arr.append(C_vals)
            P_arr.append(P_vals)

        C_arr.reverse()
        P_arr.reverse()

        option.call_p = C_vals.item()
        option.put_p = P_vals.item()

        if plot:
            arrays = [self.P_arr, self.C_arr, self.S_arr]
            dot = Digraph(name="ThreeTrees")
            dot.attr(rankdir="LR")
            names = ["Put Price", "Call Price", "Stock Call"]

            for k, array in enumerate(arrays):
                with dot.subgraph(name=f"cluster_{k}") as sub:
                    sub.attr(label=names[k])
                    sub.attr(rankdir="TB")
                    
                    # plot nodes
                    for layer_idx, layer in enumerate(array):
                        for node_idx, value in enumerate(layer):
                            node_id = f"T{k}_L{layer_idx}_N{node_idx}"
                            sub.node(node_id, label=f"{float(value):.2f}")
                        
                    # plot edges
                    for layer_idx in range(len(array) - 1):
                        for node_idx in range(len(array[layer_idx])):
                            src = f"T{k}_L{layer_idx}_N{node_idx}"
                            flat = f"T{k}_L{layer_idx+1}_N{node_idx}"
                            down = f"T{k}_L{layer_idx+1}_N{node_idx+1}"

                            sub.edge(src, flat)
                            sub.edge(src, down)

            img_bytes = dot.pipe(format='png')
            img = Image.open(io.BytesIO(img_bytes))
            plt.imshow(img)
            plt.axis("off")
            plt.show()
        if print: super()._print(option)

class BlackScholesPricer(_Pricer):
    def __init__(self):
        self.title = "Black-Scholes"

    def price(self, option: Option, print: bool = True):
        vol = option.sigma * np.sqrt(option.exp_t)
        d1 = (np.log(option.stock_p / option.strike_p) + (option.int_r + option.sigma**2 / 2) * option.exp_t) / vol
        d2 = d1 - vol

        option.call_p = option.stock_p * norm.cdf(d1) - option.strike_p * np.exp(-option.int_r * option.exp_t) * norm.cdf(d2)
        option.put_p = option.strike_p * np.exp(-option.int_r * option.exp_t) * norm.cdf(-d2) - option.stock_p * norm.cdf(-d1)
        
        if print: super()._print(option)