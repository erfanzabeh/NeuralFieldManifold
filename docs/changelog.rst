Changelog
=========

v0.1.0 (2026-02-15)
--------------------

Initial release.

**Generators:**

- TVAR coefficient schedules: sinusoid, fourier, quasiperiodic, polynomial drift, logistic transition, multi-sigmoid, Gaussian bumps, smooth random
- Time-Varying VAR(2) with rotating coupling matrices (TVVAR)
- Lorenz system (RK4 integration)
- Ornstein-Uhlenbeck process (exact analytical & Euler-Maruyama)
- Colored noise family: white, pink (1/f), brown (1/f²), arbitrary spectral exponent

**Embedders:**

- Delay embedding for state-space reconstruction

**Models:**

- AR: classical OLS autoregressive model with AIC/BIC selection
- TAR: two-regime threshold autoregressive model with bootstrap simulation
- DeepLagEmbed: geometry-aware network for joint AR-order and time-varying coefficient estimation
- ARMLP: MLP baseline for time-varying coefficient estimation
- ARTransformer: Transformer encoder baseline for time-varying coefficient estimation
- AnalyticalAR: non-learned least-squares AR baseline (PyTorch)

**Utilities:**

- Composite loss functions: classification (CE), AR reconstruction (MSE), energy constraint, smoothness penalty, order regulariser
- Training loop with ReduceLROnPlateau scheduling and gradient clipping
- Benchmark loop with per-order accuracy, coefficient MSE, and signal MSE
- Bicoherence for quadratic phase-coupling detection

**Plotting:**

- Training history dashboard (2x6 grid with confusion matrix)
- Per-order coefficient visualisation (true vs. predicted)
- TVAR sample inspector (coefficients, signal, sliding-window power)
