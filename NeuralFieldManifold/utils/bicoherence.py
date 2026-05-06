import numpy as np
from typing import Optional


def bicoherence(
    x: np.ndarray,
    fs: float,
    nperseg: int = 256,
    noverlap: Optional[int] = None,
    fmax: Optional[float] = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute bicoherence for detecting quadratic phase coupling.
    
    Bicoherence values near 1 indicate strong phase-locking between 
    frequency components f1, f2, and f1+f2.
    
    Parameters
    ----------
    x : np.ndarray
        Input signal (1D).
    fs : float
        Sampling frequency in Hz.
    nperseg : int
        Segment length for FFT. Default is 256.
    noverlap : int, optional
        Overlap between segments. Default is nperseg // 2.
    fmax : float, optional
        Maximum frequency to compute. Default is fs / 4.
    
    Returns
    -------
    freqs : np.ndarray
        Frequency axis.
    bic : np.ndarray
        Bicoherence matrix (f1 x f2).
    
    Example
    -------
    >>> freqs, bic_matrix = bicoherence(signal, fs=500, nperseg=512, fmax=50)
    """
    if noverlap is None:
        noverlap = nperseg // 2
    if fmax is None:
        fmax = fs / 4
    
    # Segment signal
    step = nperseg - noverlap
    n_seg = (len(x) - nperseg) // step + 1
    
    # Frequency resolution
    freqs = np.fft.rfftfreq(nperseg, 1 / fs)
    mask = freqs <= fmax
    freqs_out = freqs[mask]
    nf = len(freqs_out)
    
    # Accumulators
    bispectrum = np.zeros((nf, nf), dtype=complex)
    norm1 = np.zeros((nf, nf))
    norm2 = np.zeros((nf, nf))
    
    window = np.hanning(nperseg)
    
    for i in range(n_seg):
        seg = x[i * step : i * step + nperseg]
        seg = (seg - seg.mean()) * window
        
        X = np.fft.rfft(seg)
        X_trunc = X[:nf]
        
        for i1 in range(nf):
            for i2 in range(i1, nf):
                i3 = i1 + i2
                if i3 < len(X):
                    prod = X_trunc[i1] * X_trunc[i2] * np.conj(X[i3])
                    bispectrum[i1, i2] += prod
                    norm1[i1, i2] += np.abs(X_trunc[i1] * X_trunc[i2]) ** 2
                    norm2[i1, i2] += np.abs(X[i3]) ** 2
    
    # Compute bicoherence
    with np.errstate(divide='ignore', invalid='ignore'):
        bic = np.abs(bispectrum) ** 2 / (norm1 * norm2 + 1e-12)
    
    # Mirror to lower triangle
    bic = bic + bic.T - np.diag(np.diag(bic))
    
    return freqs_out, bic
