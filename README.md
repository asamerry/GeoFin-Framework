# Finance-Engine

## Welcome

This is a modular portfolio optimization framework built for systematic asset allocation and quantitative experimentation. The application combines configurable return estimators, risk models, and convex optimization techniques to construct optimal portfolios under user-defined constraints. Data is pulled using the `yfinance` Python API and cached locally for efficiency, while model behavior is fully controlled through YAML configuration files. Whether using classical Markowitz optimization or looking for more advance techniques, this application is designed to provide a clean, extensible foundation for exploring modern portfolio theory and building more advanced quantitative strategies.

Optimizers:
- Markowitz

Return Estimators:
- Historic
- CAPM
- Black-Litterman

Risk Estimators:
- Variance

We've provided output options as well for those that want to view the graph of the Efficient Frontier or want to save their optimized portfolios. 


## Getting Started

To use this application, simply open your command terminal and run the following commands:
```bash
$ git clone https://github.com/asamerry/Finance-Engine.git
$ cd Finance-Engine
$ make
```

Next, you're going to want to create a configuration file for the application. We have provided an example configuration file at `configs/config.yaml` to help you get started. If you're using Black-Litterman return estimates, you'll also want to create a views file. An example views file has been provided as `views/views.txt`. 

All that's left to do is run the application using 
```bash
$ python3 main.py --config [path-to-config-file]
```

Prices and market cap data will automatically be cached daily to speed up successive runs, but you can force new data by including the `--recache` tag. 


## About the Optimizers

### Markowitz Optimizer

Also known as the Mean–Variance optimizer, the model aims to minimize portfolio risk for a specified target return. For each feasible return level within the range of estimated expected returns, we solve a constrained quadratic optimization problem and record the corresponding optimal portfolio. The collection of these optimal portfolios forms the efficient frontier — the set of portfolios offering the highest expected return for each level of risk.

From this frontier, the application selects the portfolio with the maximum Sharpe ratio, defined as excess return relative to risk. This portfolio corresponds to the tangency point between the efficient frontier and the capital allocation line implied by the risk-free rate.

Mathematically, the optimization problem is formulated as:

$$\begin{align*}
      \text{minimize}\ \ & \underline{\omega}^T \underline{\underline{\Sigma}} \underline{\omega} + \lambda \gamma(\underline{\omega}) \\
    \text{subject to}\ \ & \underline{\mu^T} \underline{\omega} = r \\
                         & \underline{1}^T \underline{\omega} = 1
\end{align*}$$

where $\underline{\omega}$ are the weights of your portfolio, $\underline{\mu}$ is the expected returns vector, $\underline{\underline{\Sigma}}$ is the risk matrix of the returns, $r$ is the target return rate, and $\gamma$ is an optional penalty function with parameter $\lambda$. Additionally, we can add a constraint for non-negativity, $\underline{\omega} \geq 0$, to restrict the model to long investments only. 

Since the objective is quadratic and the constraints are linear, the problem is solved using convex quadratic programming. This guarantees global optimality when the covariance matrix is positive semi-definite.

While conceptually simple, the Mean–Variance framework remains foundational in modern portfolio theory. Its structure provides a clear tradeoff between risk and return, and it serves as the basis for many more advanced models, included some of those implemented within this framework. 


## About the Return Estimators

### Historic Returns

The simplest and most direct method of estimating expected returns is by use of historical averages of asset returns. This estimator assumes that past return behavior is informative about future expectations and serves as the baseline return model. 

The formula is given as

$$\underline{\mu} := \mathbb E[\underline{R}] = \frac1T \sum_{t=1}^T \frac{P_{t} - P_{t-1}}{P_{t-1}},$$

where $P_t$ is the price of an asset at time $t$. 

While simple, historical estimation is fully data-driven, avoids imposing equilibrium assumptions, and provides a strong basis for the construction of more complex return estimations. However, it may be sensitive to sampling noise, regime shifts, and limited data windows.

### CAPM Returns

The Capital Asset Pricing Model (CAPM) provides a method for estimating expected returns based on systematic market risk. Rather than relying solely on historical averages, CAPM links expected returns to an asset’s exposure to the overall market.

The CAPM formula is given by

$$\underline{\mu} := \mathbb E[\underline{R}] = r_f + \underline{\beta}(r_m - r_f),$$

where $r_f$ is the market risk-free rate, $r_m$ is the equilibrium market return, and $\underline{\beta}$ is the vector of systematic risk for each portfolio asset. 

CAPM-based estimates are particularly useful when seeking theoretically grounded return forecasts or when historical means appear unstable.

### Black-Litterman Returns

The Black-Litterman Model provides a more advanced architecture for estimating returns based not only on prior returns, but also on the personal views of the investor. The model was developed by Fisher Black and Robert Litterman in 1990 at Goldman Sachs whose goal was to allow investors to provided subjective information to a rigorously mathematical model. 

The expected returns are given as follows:

$$\underline{\mu} := \mathbb E[\underline{R}] = [(\tau\underline{\underline{\Sigma}})^{-1} + \underline{\underline{P}}^T\underline{\underline{\Omega}}^{-1}\underline{\underline{P}}]^{-1}[(\tau\underline{\underline{\Sigma}})^{-1}\underline{\Pi} + \underline{\underline{P}}^T\underline{\underline{\Omega}}^{-1}\underline{Q}],$$

where $\tau$ is a scalar, $\underline{\underline{\Sigma}}$ is the risk matrix of returns, $\underline{Q}$ is the vector of view weights, $\underline{\underline{P}}$ is the picking matrix of views, $\underline{\underline{\Omega}}$ is the uncertainty matrix of views, and $\underline{\Pi}$ is the vector of prior expected returns. 

When using Black-Litterman returns in this application, we requires views files with a list of views, written in a spcific, interpretable format. There are two ways to state a view. 
1. Absolute views: `ASSET %change [up/down]`
2. Relative views: `ASSET1 %change [over/under] ASSET2`

For example, if you think that the price of Amazon stock will rise by 10%, we write `AMZN 0.1 up`. If you think that the price of Apple will rise by 20% relative to Microsoft, then we write `AAPL 0.2 over MSFT`.


## About the Risk Estimators

### Variance

Variance is the simplest and most intuitive of the risk measures available to us. Specifically, we uses the covariance matrix of asset returns, denoted as $\underline{\underline{\Sigma}}$. 

The formula is 

$$\text{Risk}(\underline{\omega}) = \text{Var}(\underline{R} \underline{\omega}) = \underline{\omega}^T\underline{\underline{\Sigma}} \underline{\omega}.$$

This quadratic form captures both individual asset volatility and cross-asset correlations. Diversification effects arise naturally from the covariance structure: assets with low or negative correlations reduce overall portfolio variance.

In this application, the covariance matrix is estimated directly from historical return data. Because the resulting matrix is positive semi-definite, the associated optimization problem remains convex and globally solvable.

Variance-based risk is the classical measure used in Modern Portfolio Theory and forms the foundation of the Markowitz framework.
