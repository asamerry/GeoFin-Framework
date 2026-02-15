# GeoFin-Framework

## Welcome

This is a modular quantitative finance framework built for systematic asset allocation, derivatives pricing, and computational experimentation. The application integrates configurable portfolio optimization models with classical options pricing techniques, combining modern portfolio theory, equilibrium-based return estimation, and no-arbitrage derivative valuation within a unified architecture.

For portfolio construction, the framework supports convex optimization models driven by customizable return and risk estimators. For derivatives, it implements both discrete-time and continuous-time pricing models, providing numerical and closed-form valuation methods. Market data is retrieved using the `yfinance` Python API and cached locally for efficiency, while all model behavior is controlled through structured YAML configuration files.

The framework currently includes the following models:

1. Portfolio Optimization Models:
    - Markowitz 

    Return Estimators:
    - Historic
    - CAPM
    - Black-Litterman

    Risk Estimators:
    - Variance

  2. Options Pricing Models:
     - Binomial
     - Black-Scholes


## Getting Started

To use this application, simply open your command terminal and run the following commands:
```bash
$ git clone https://github.com/asamerry/GeoFin-Framework.git
$ cd GeoFin-Framework
$ make
```

Next, you're going to want to create a configuration file for the application. We have provided an example configuration file at `configs/config.yaml` to help you get started. If you're using Black-Litterman return estimates, you'll also want to create a views file. An example views file has been provided as `views/views.txt`. 

All that's left to do is run the application using 
```bash
$ python3 main.py --config [path-to-config-file] -t [portfolio-optimization/options-pricing]
```

Prices and market cap data will automatically be cached daily to speed up successive runs, but you can force new data by including the `--recache` tag. 


## Portfolio Optimization Models

### About the Optimizers

#### Markowitz Optimizer

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


### About the Return Estimators

#### Historic Returns

The simplest and most direct method of estimating expected returns is by use of historical averages of asset returns. This estimator assumes that past return behavior is informative about future expectations and serves as the baseline return model. 

The formula is given as

$$\underline{\mu} := \mathbb E[\underline{R}] = \frac1T \sum_{t=1}^T \frac{P_{t} - P_{t-1}}{P_{t-1}},$$

where $P_t$ is the price of an asset at time $t$. 

While simple, historical estimation is fully data-driven, avoids imposing equilibrium assumptions, and provides a strong basis for the construction of more complex return estimations. However, it may be sensitive to sampling noise, regime shifts, and limited data windows.

#### CAPM Returns

The Capital Asset Pricing Model (CAPM) provides a method for estimating expected returns based on systematic market risk. Rather than relying solely on historical averages, CAPM links expected returns to an asset’s exposure to the overall market.

The CAPM formula is given by

$$\underline{\mu} := \mathbb E[\underline{R}] = r_f + \underline{\beta}(r_m - r_f),$$

where $r_f$ is the market risk-free rate, $r_m$ is the equilibrium market return, and $\underline{\beta}$ is the vector of systematic risk for each portfolio asset. 

CAPM-based estimates are particularly useful when seeking theoretically grounded return forecasts or when historical means appear unstable.

#### Black-Litterman Returns

The Black-Litterman Model provides a more advanced architecture for estimating returns based not only on prior returns, but also on the personal views of the investor. The model was developed by Fisher Black and Robert Litterman in 1990 at Goldman Sachs whose goal was to allow investors to provided subjective information to a rigorously mathematical model. 

The expected returns are given as follows:

$$\underline{\mu} := \mathbb E[\underline{R}] = [(\tau\underline{\underline{\Sigma}})^{-1} + \underline{\underline{P}}^T\underline{\underline{\Omega}}^{-1}\underline{\underline{P}}]^{-1}[(\tau\underline{\underline{\Sigma}})^{-1}\underline{\Pi} + \underline{\underline{P}}^T\underline{\underline{\Omega}}^{-1}\underline{Q}],$$

where $\tau$ is a scalar, $\underline{\underline{\Sigma}}$ is the risk matrix of returns, $\underline{Q}$ is the vector of view weights, $\underline{\underline{P}}$ is the picking matrix of views, $\underline{\underline{\Omega}}$ is the uncertainty matrix of views, and $\underline{\Pi}$ is the vector of prior expected returns. 

When using Black-Litterman returns in this application, we requires views files with a list of views, written in a spcific, interpretable format. There are two ways to state a view. 
1. Absolute views: `ASSET %change [up/down]`
2. Relative views: `ASSET1 %change [over/under] ASSET2`

For example, if you think that the price of Amazon stock will rise by 10%, we write `AMZN 0.1 up`. If you think that the price of Apple will rise by 20% relative to Microsoft, then we write `AAPL 0.2 over MSFT`.


### About the Risk Estimators

#### Variance

Variance is the simplest and most intuitive of the risk measures available to us. Specifically, we uses the covariance matrix of asset returns, denoted as $\underline{\underline{\Sigma}}$. 

The formula is 

$$\text{Risk}(\underline{\omega}) = \text{Var}(\underline{R} \underline{\omega}) = \underline{\omega}^T\underline{\underline{\Sigma}} \underline{\omega}.$$

This quadratic form captures both individual asset volatility and cross-asset correlations. Diversification effects arise naturally from the covariance structure: assets with low or negative correlations reduce overall portfolio variance.

In this application, the covariance matrix is estimated directly from historical return data. Because the resulting matrix is positive semi-definite, the associated optimization problem remains convex and globally solvable.

Variance-based risk is the classical measure used in Modern Portfolio Theory and forms the foundation of the Markowitz framework.


## Options Pricing Models

### About the Pricers

#### Binomial Pricing

The Binomial Pricing Model provides a versatile, discrete-time approach to options pricing. The model operates by constructing a binomial decision tree under the assumption that the price of the underlying asset can only move up ($u = e^{(r-\delta)h + \sigma\sqrt h}$) or down ($d = e^{(r-\delta)h - \sigma\sqrt h}$) at each node, allowing for the pricing of European, American, and exotic options. 

This model prices options by defining $p = \frac{e^{(r-\delta)h} - d}{u - d}$ to iteratively solve backward for 

$$C(S_0, h) = e^{-rh}(pC_u + (1-p)C_d) \quad \text{and} \quad P(S_0, h) = e^{-rh}(pP_u + (1-p)P_d),$$

where $h$ is the step-size, $r$ the interest rate, and $\delta$ the dividend return rate. 

As the step size approaches zero, the Binomial Model approachs the Black-Scholes Model, providing a bridge between discrete and continuous pricing methods. However, the tree structure of the Binomial model allows easy extension to American options, and we have provided functionality to plot the tree graph for analysis of intermediary layers. 

#### Black-Scholes Pricing

The Black-Scholes Model provides a closed-form solution for pricing European call options under the assumption that the underlying asset following a geometric Brownian motion with constant volatility. 

Under this model, European call and put options are priced by

$$C(S_0, T) = S_0N(d_1) - Ke^{-rT}N(d_2) \quad \text{and} \quad P(S_0, T) = Ke^{-rT}N(-d_2) - S_0N(-d_1),$$

where $S_0$ is the spot price of the underlying asset, $K$ is the strike price of the contract, $r$ is the interest rate, $T$ is the time to expiry, $N$ is the Gaussian cdf, 

$$d_1 = \frac{\ln(\frac{S_0}{K}) + (r + \frac{\sigma^2}2)T}{\sigma\sqrt T}, \quad \text{and} \quad d_2 = \frac{\ln(\frac{S_0}{K}) + (r - \frac{\sigma^2}2)T}{\sigma\sqrt T}.$$

The Black-Scholes arises from the construction of a riskless replicating portfolio and solving the associated parabolic PDE under no-arbitrage conditions. Despite its simplifying assumptions and restriction to European options, the model provides a fondational framework for more advanced models such as the Stochastic Volatility and Jump Diffusion models. 
