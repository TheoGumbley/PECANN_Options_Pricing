# PECANN for Options Pricing

This code trains a Physics and Equality Constrained Artificial Neural Network (PECANN) to price European call options under a partially non-dimensional Black-Scholes model, implemented in Python (PyTorch). The network is trained across a continuous range of asset prices, times, risk-free rates, and volatilities, producing a single model valid for a whole family of options rather than one fixed strike/volatility combination.

A full technical write-up of the theory, methodology, and results is included in `PECANN\_Technical\_Doc\_final.pdf`.

\---

## Methods

**Non-dimensionalisation**

* Asset price, option value, and time normalised by strike price K and time to maturity T
* Network takes 4 inputs: non-dimensional asset price Ŝ, non-dimensional time t̂, risk-free rate r, volatility σ

**Loss formulation**

|Formulation|Boundary/Terminal/Data Losses|Convergence Behaviour|
|-|-|-|
|Standard PINN|Soft, equally-weighted MSE terms|Disparate loss magnitudes distort the loss surface|
|PECANN (this project)|Hard equality constraints via Augmented Lagrangian Method (ALM)|PDE residual minimised directly; BC/data losses driven toward zero|

**Optimiser strategy**

* Adam (lr=1e-3, weight decay=1e-5) for the first `OPTIMIZER\_SWITCH\_EPOCH` epochs
* LBFGS with Strong Wolfe line search for the remainder of training

**Adaptive collocation refinement (optional)**

* KDTree-based nearest-neighbour search identifies high-residual regions
* New collocation points sampled within a hypersphere around each violating point using Gaussian-normalised direction vectors and a volumetrically-correct radius
* Controlled via `adapt\_params`: refinement start epoch, residual threshold, max collocation points, points added per adaptation, adaptation period

**Validation**

* RMSE computed against the closed-form Black-Scholes call price over a 200-point Ŝ grid, at low/mid/high (σ, r) combinations
* `seed\_data\_analytics` repeats training over multiple random seeds to assess run-to-run variance (RMSE violin plots + summary statistics)

\---

## Known limitations

* Restricted to European-style call options with a fixed one-year maturity (T is not yet a network input)
* Accuracy is weakest near the terminal payoff (t̂ → 1), where the smooth tanh activation struggles to represent the kink in the payoff function
* Constant volatility is assumed; extensions such as Heston volatility are not implemented

See the technical document's Debugging \& Design Decisions and Future Work sections for further discussion.

\---

## Dependencies

```
numpy
scipy
matplotlib
torch
```

Install with:

```bash
pip install numpy scipy matplotlib torch
```

This script also imports a local `Black\_Scholes.py` module (closed-form call pricing functions), which must be present in the same directory.

\---

## Usage

Configuration constants (data/collocation point counts, hidden layer width, epoch counts, optimiser switch point) are set near the top of `pecann\_black\_scholes.py`. PECANN hyperparameters (`pecann\_params`) and adaptive refinement hyperparameters (`adapt\_params`) are set as dictionaries near the bottom of the file:

```python
pecann\_params = {
    "k": 50,          # Lagrange multiplier / penalty update period (epochs)
    "beta": 1.05,     # penalty growth rate
    "gamma": 0.5,     # residual ratio threshold
    "max mu": 1000,   # maximum penalty parameter
    "starting epoch": 0
}

adapt\_params = {
    "starting epoch": 5000,
    "residual threshold": 5e-5,
    "maximum collocation points": 25000,
    "per adaptation added points": 3,
    "adaptation period": 500,
}
```

Run with:

```bash
python PECANN\_FINAL.py
```

By default this runs `seed\_data\_analytics`, which trains the model 10 times with different random seeds and reports RMSE mean/standard deviation and violin plots across low/mid/high (σ, r) parameter regimes. This is computationally expensive (multiple hours depending on hardware). To run a single training pass instead, call `train(...)` and `validate(...)` directly with the output of `generate\_data()`.

The script outputs:

* PECANN loss curves (data, PDE residual, boundary condition) vs epoch
* Evolution of the ALM penalty parameters μ and residual ratios during training
* Payoff curve vs the closed-form solution at present day
* Absolute error to the closed-form solution across the Ŝ domain
* RMSE vs t̂
* (if `adapt=True`) collocation point residual and refinement diagnostic plots
* RMSE summary statistics and violin plots across repeated training runs

\---

## Author

Theo Gumbley — Aerospace \& CFD Engineer, Southampton University

