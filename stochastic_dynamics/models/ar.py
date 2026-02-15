import numpy as np

class AR:
    """Classical Autoregressive (AR) model fitted via ordinary least squares."""

    @staticmethod
    def lag_matrix(x, p):
        """Construct the lag (design) matrix and target vector for AR(p).

        Parameters
        ----------
        x : np.ndarray
            1-D time series of length *N*.
        p : int
            Autoregressive order.

        Returns
        -------
        X : np.ndarray
            Lag matrix of shape ``(N - p, p)`` where column *i* contains
            the lag-(i+1) values.
        y : np.ndarray
            Target vector of shape ``(N - p,)``.
        """
        N = len(x)
        y = x[p:].copy()
        X = np.zeros((N-p, p))
        for i in range(p):
            X[:, i] = x[p-1-i : N-1-i]
        return X, y

    @staticmethod
    def fit(X, y):
        """Fit AR coefficients via OLS with an intercept term.

        Parameters
        ----------
        X : np.ndarray
            Lag matrix of shape ``(N, p)``.
        y : np.ndarray
            Target vector of shape ``(N,)``.

        Returns
        -------
        w : np.ndarray
            Coefficient vector ``[c, a1, …, ap]`` of shape ``(p + 1,)``
            where *c* is the intercept.
        """
        D = np.column_stack([np.ones(len(X)), X])
        w = np.linalg.lstsq(D, y, rcond=None)[0]
        return w  # [c, a1..ap]

    @staticmethod
    def predict_from_params(X, w):
        """Predict target values from lag matrix and fitted parameters.

        Parameters
        ----------
        X : np.ndarray
            Lag matrix of shape ``(N, p)``.
        w : np.ndarray
            Coefficient vector ``[c, a1, …, ap]`` of shape ``(p + 1,)``.

        Returns
        -------
        y_hat : np.ndarray
            Predicted values of shape ``(N,)``.
        """
        D = np.column_stack([np.ones(len(X)), X])
        return D @ w

    @staticmethod
    def metrics(y_true, y_pred):
        """Compute regression metrics between true and predicted signals.

        Parameters
        ----------
        y_true : np.ndarray
            Ground-truth values.
        y_pred : np.ndarray
            Predicted values.

        Returns
        -------
        mse : float
            Mean squared error.
        mae : float
            Mean absolute error.
        r : float
            Pearson correlation coefficient.
        """
        mse = np.mean((y_true - y_pred)**2)
        mae = np.mean(np.abs(y_true - y_pred))
        yt = y_true - np.mean(y_true)
        yp = y_pred - np.mean(y_pred)
        r = (yt @ yp) / np.sqrt((yt @ yt) * (yp @ yp))
        return mse, mae, r

    @staticmethod
    def acf(sig, nlags=60):
        """Compute the normalized autocorrelation function.

        Parameters
        ----------
        sig : np.ndarray
            Input signal.
        nlags : int, optional
            Number of lags to return. Default is 60.

        Returns
        -------
        ac : np.ndarray
            Autocorrelation values from lag 0 to *nlags*, inclusive.
        """
        s = sig - np.mean(sig)
        ac = np.correlate(s, s, mode="full")
        ac = ac[ac.size//2:]
        ac0 = ac[0] if ac[0] != 0 else 1.0
        ac = ac / ac0
        return ac[:nlags+1]

    @staticmethod
    def aic_bic(y, yhat, k):
        """Compute AIC and BIC for an AR model.

        Parameters
        ----------
        y : np.ndarray
            Observed values.
        yhat : np.ndarray
            Predicted values.
        k : int
            Number of estimated parameters (including intercept).

        Returns
        -------
        aic : float
            Akaike Information Criterion.
        bic : float
            Bayesian Information Criterion.
        """
        n = len(y)
        rss = np.sum((y - yhat)**2)
        sigma2 = rss / max(n, 1)
        aic = n*np.log(sigma2 + 1e-12) + 2*k
        bic = n*np.log(sigma2 + 1e-12) + k*np.log(max(n, 2))
        return aic, bic
    
    @staticmethod
    def hybrid_predict(series, w, p, start_idx, n_steps, refresh_every=1):
        """Multi-step ahead prediction with periodic history refresh.

        Runs a free-running AR forecast but resets the lag buffer to the
        true signal every *refresh_every* steps, blending open-loop and
        closed-loop prediction.

        Parameters
        ----------
        series : np.ndarray
            Full observed time series.
        w : np.ndarray
            AR coefficient vector ``[c, a1, …, ap]``.
        p : int
            Autoregressive order.
        start_idx : int
            Index in *series* at which prediction begins.
        n_steps : int
            Number of steps to forecast.
        refresh_every : int, optional
            Re-anchor the lag buffer to the true signal every this many
            steps. Default is 1 (teacher-forced).

        Returns
        -------
        preds : np.ndarray
            Predicted values of shape ``(n_steps,)``.
        """
        preds = []
        hist = series[start_idx - p : start_idx].tolist()
        for t in range(n_steps):
            if (t % max(int(refresh_every), 1)) == 0:
                abs_idx = start_idx + t
                hist = series[abs_idx - p : abs_idx].tolist()
            xhat = w[0] + np.dot(w[1:], list(reversed(hist)))
            preds.append(xhat)
            hist.pop(0)
            hist.append(xhat)
        return np.array(preds)