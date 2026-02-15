import numpy as np

# Shared lag structure: gives each lag a distinct identity
_BASE_FREQS   = np.array([1/150, 1/200, 1/180, 1/220, 1/170, 1/190, 1/160, 1/210, 1/175, 1/195])
_BASE_AMPS    = np.array([0.35, 0.25, 0.20, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06])
_BASE_OFFSETS = np.array([0.6, -0.5, 0.3, -0.2, 0.15, -0.1, 0.08, -0.06, 0.05, -0.04])
_BASE_PHASES  = np.array([k * np.pi / 4 for k in range(10)])

def _lag_params(p, l1_target=0.85):
    """Return per-lag frequency, amplitude, offset, and phase vectors.

    Scales the base templates so that the worst-case L1 norm
    ``sum(|offset_k| + |amp_k|)`` stays at *l1_target* (with 20 %
    jitter headroom), keeping the AR process stable by construction.

    Parameters
    ----------
    p : int
        Number of lags.
    l1_target : float, optional
        Target worst-case L1 coefficient norm. Default is 0.85.

    Returns
    -------
    freqs : np.ndarray
        Base frequencies of shape ``(p,)``.
    amps : np.ndarray
        Scaled amplitudes of shape ``(p,)``.
    offsets : np.ndarray
        Scaled offsets of shape ``(p,)``.
    phases : np.ndarray
        Phase offsets of shape ``(p,)``.
    """
    freqs   = _BASE_FREQS[:p].copy()
    amps    = _BASE_AMPS[:p].copy()
    offsets = _BASE_OFFSETS[:p].copy()
    phases  = _BASE_PHASES[:p].copy()

    # Worst-case L1: every lag at its peak, with ~20% jitter headroom
    jitter_headroom = 1.20
    raw_l1 = np.sum(np.abs(offsets) + np.abs(amps)) * jitter_headroom

    scale = l1_target / raw_l1 if raw_l1 > 0 else 1.0

    return freqs, amps * scale, offsets * scale, phases


def sinusoid(T, p, rng, freq_jitter=0.15, amp_jitter=0.15):
    """Sinusoidal coefficient schedule with per-lag variation.

    Each lag gets a distinct base frequency, amplitude, and offset
    with small random jitter for stochastic diversity across samples.

    Parameters
    ----------
    T : int
        Length of the time series.
    p : int
        AR order (number of lags).
    rng : np.random.Generator
        NumPy random generator instance.
    freq_jitter : float, optional
        Relative frequency jitter range. Default is 0.15.
    amp_jitter : float, optional
        Relative amplitude jitter range. Default is 0.15.

    Returns
    -------
    a : np.ndarray
        Time-varying AR coefficients of shape ``(T, p)``.
    """
    t = np.arange(T)
    freqs, amps, offsets, phases = _lag_params(p)
    
    # Add small random jitter to keep stochasticity across samples
    freqs  *= (1 + rng.uniform(-freq_jitter, freq_jitter, size=p))
    amps   *= (1 + rng.uniform(-amp_jitter, amp_jitter, size=p))
    offsets *= (1 + rng.uniform(-0.1, 0.1, size=p))
    phases += rng.uniform(-np.pi/8, np.pi/8, size=p)
    
    a = np.zeros((T, p), dtype=np.float32)
    for k in range(p):
        if k % 2 == 0:
            a[:, k] = offsets[k] + amps[k] * np.sin(2*np.pi*freqs[k]*t + phases[k])
        else:
            a[:, k] = offsets[k] + amps[k] * np.cos(2*np.pi*freqs[k]*t + phases[k])
    return a


def fourier(T, p, rng, M=3, amp_jitter=0.2):
    """Fourier-series coefficient schedule with decaying harmonics.

    Each lag is represented by a truncated Fourier series with *M*
    harmonics whose amplitudes decay as ``1 / m^1.2``.

    Parameters
    ----------
    T : int
        Length of the time series.
    p : int
        AR order (number of lags).
    rng : np.random.Generator
        NumPy random generator instance.
    M : int, optional
        Number of Fourier harmonics per lag. Default is 3.
    amp_jitter : float, optional
        Relative amplitude jitter per harmonic. Default is 0.2.

    Returns
    -------
    a : np.ndarray
        Time-varying AR coefficients of shape ``(T, p)``.
    """
    t = np.arange(T)
    freqs, amps, offsets, phases = _lag_params(p)
    freqs *= (1 + rng.uniform(-0.15, 0.15, size=p))
    
    a = np.zeros((T, p), dtype=np.float32)
    for k in range(p):
        ak = offsets[k] * np.ones(T)
        for m in range(1, M+1):
            harm_amp = amps[k] / (m**1.2) * (1 + rng.uniform(-amp_jitter, amp_jitter))
            phi = phases[k] * m + rng.uniform(-np.pi/6, np.pi/6)
            if (k + m) % 2 == 0:
                ak += harm_amp * np.sin(2*np.pi*(m*freqs[k])*t + phi)
            else:
                ak += harm_amp * np.cos(2*np.pi*(m*freqs[k])*t + phi)
        a[:, k] = ak
    return a


def quasiperiodic(T, p, rng, freq_jitter=0.15):
    """Quasiperiodic coefficient schedule with two incommensurate frequencies.

    Each lag combines two sinusoidal components whose frequency ratio
    is irrational, producing non-repeating oscillatory patterns.

    Parameters
    ----------
    T : int
        Length of the time series.
    p : int
        AR order (number of lags).
    rng : np.random.Generator
        NumPy random generator instance.
    freq_jitter : float, optional
        Relative frequency jitter range. Default is 0.15.

    Returns
    -------
    a : np.ndarray
        Time-varying AR coefficients of shape ``(T, p)``.
    """
    t = np.arange(T)
    freqs, amps, offsets, phases = _lag_params(p)
    
    # Second set of frequencies (shifted)
    freqs2 = np.roll(freqs, 1) * 1.3 * (1 + rng.uniform(-freq_jitter, freq_jitter, size=p))
    amps2 = amps * rng.uniform(0.4, 0.8, size=p)
    phases2 = phases + rng.uniform(np.pi/4, np.pi/2, size=p)
    
    freqs *= (1 + rng.uniform(-freq_jitter, freq_jitter, size=p))
    
    a = np.zeros((T, p), dtype=np.float32)
    for k in range(p):
        a[:, k] = (offsets[k] 
                    + amps[k] * np.sin(2*np.pi*freqs[k]*t + phases[k])
                    + amps2[k] * np.cos(2*np.pi*freqs2[k]*t + phases2[k]))
    return a


def polynomial_drift(T, p, rng, degree=3, coef_jitter=0.3):
    """Polynomial-drift coefficient schedule.

    Each lag's trajectory is a random polynomial of the given degree
    centred on the lag-specific offset.

    Parameters
    ----------
    T : int
        Length of the time series.
    p : int
        AR order (number of lags).
    rng : np.random.Generator
        NumPy random generator instance.
    degree : int, optional
        Polynomial degree. Default is 3.
    coef_jitter : float, optional
        Relative jitter on polynomial coefficients. Default is 0.3.

    Returns
    -------
    a : np.ndarray
        Time-varying AR coefficients of shape ``(T, p)``.
    """
    tn = np.linspace(-1, 1, T)
    _, amps, offsets, _ = _lag_params(p)
    
    a = np.zeros((T, p), dtype=np.float32)
    for k in range(p):
        # Start from the structured offset
        ak = offsets[k] * np.ones(T)
        # Add polynomial variation scaled by lag amplitude
        betas = rng.normal(0, amps[k] * (1 + rng.uniform(-coef_jitter, coef_jitter)), size=degree+1)
        betas[0] = 0  # offset already handled
        for d in range(1, degree+1):
            ak += betas[d] * (tn**d)
        a[:, k] = ak
    return a


def logistic_transition(T, p, rng):
    """Single logistic-transition coefficient schedule.

    Each lag transitions between two distinct levels via a sigmoid
    whose centre and steepness are randomised.

    Parameters
    ----------
    T : int
        Length of the time series.
    p : int
        AR order (number of lags).
    rng : np.random.Generator
        NumPy random generator instance.

    Returns
    -------
    a : np.ndarray
        Time-varying AR coefficients of shape ``(T, p)``.
    """
    tau_range = (T/10, T/4)
    t = np.arange(T)
    _, amps, offsets, _ = _lag_params(p)
    
    a = np.zeros((T, p), dtype=np.float32)
    for k in range(p):
        # Transition between offset +/- amp (distinct per lag)
        alo = offsets[k] - amps[k] * (1 + rng.uniform(-0.2, 0.2))
        ahi = offsets[k] + amps[k] * (1 + rng.uniform(-0.2, 0.2))
        t0  = rng.integers(int(0.2*T), int(0.8*T))
        tau = rng.uniform(tau_range[0], tau_range[1])
        s = 1.0 / (1.0 + np.exp(-(t - t0)/tau))
        a[:, k] = alo + (ahi - alo) * s
    return a


def multi_sigmoid(T, p, rng, J=3):
    """Multi-sigmoid step coefficient schedule.

    Superimposes *J* sigmoid transitions of random magnitude and
    steepness around each lag's baseline offset.

    Parameters
    ----------
    T : int
        Length of the time series.
    p : int
        AR order (number of lags).
    rng : np.random.Generator
        NumPy random generator instance.
    J : int, optional
        Number of sigmoid steps per lag. Default is 3.

    Returns
    -------
    a : np.ndarray
        Time-varying AR coefficients of shape ``(T, p)``.
    """
    tau_range = (T/20, T/8)
    t = np.arange(T)
    _, amps, offsets, _ = _lag_params(p)
    
    a = np.zeros((T, p), dtype=np.float32)
    for k in range(p):
        ak = offsets[k] * np.ones(T)
        t0s = np.sort(rng.integers(int(0.1*T), int(0.9*T), size=J))
        for j in range(J):
            delta = rng.normal(0, amps[k] * 0.5)
            tau = rng.uniform(tau_range[0], tau_range[1])
            s = 1.0 / (1.0 + np.exp(-(t - t0s[j])/tau))
            ak += delta * s
        a[:, k] = ak
    return a


def gaussian_bumps(T, p, rng, J=4):
    """Gaussian-bump coefficient schedule.

    Overlays *J* Gaussian-shaped bumps of random centre, width, and
    sign on each lag's baseline offset.

    Parameters
    ----------
    T : int
        Length of the time series.
    p : int
        AR order (number of lags).
    rng : np.random.Generator
        NumPy random generator instance.
    J : int, optional
        Number of Gaussian bumps per lag. Default is 4.

    Returns
    -------
    a : np.ndarray
        Time-varying AR coefficients of shape ``(T, p)``.
    """
    width_range = (T/10, T/5)
    t = np.arange(T)
    _, amps, offsets, _ = _lag_params(p)
    
    a = np.zeros((T, p), dtype=np.float32)
    for k in range(p):
        ak = offsets[k] * np.ones(T)
        for _ in range(J):
            mu = rng.uniform(0.1*T, 0.9*T)
            sig = rng.uniform(width_range[0], width_range[1])
            c = rng.uniform(-amps[k], amps[k])
            ak += c * np.exp(-0.5 * ((t - mu)/sig)**2)
        a[:, k] = ak
    return a


def smooth_random(T, p, rng):
    """Smooth random (GP-like) coefficient schedule.

    Generates spectrally shaped Gaussian noise with a long correlation
    length, producing smooth, aperiodic coefficient trajectories
    centred on each lag's baseline.

    Parameters
    ----------
    T : int
        Length of the time series.
    p : int
        AR order (number of lags).
    rng : np.random.Generator
        NumPy random generator instance.

    Returns
    -------
    a : np.ndarray
        Time-varying AR coefficients of shape ``(T, p)``.
    """
    smooth_sigma = T/10
    _, amps, offsets, _ = _lag_params(p)
    
    a = np.zeros((T, p), dtype=np.float32)
    freqs = np.fft.rfftfreq(T)
    H = np.exp(-2*(np.pi**2) * (smooth_sigma**2) * (freqs**2))
    for k in range(p):
        w = rng.normal(0, 1, size=T)
        W = np.fft.rfft(w)
        sm = np.fft.irfft(W * H, n=T)
        sm = sm / (np.std(sm) + 1e-8)
        a[:, k] = (offsets[k] + amps[k] * sm).astype(np.float32)
    return a