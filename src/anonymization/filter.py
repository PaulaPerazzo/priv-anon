import numpy as np
from scipy.fft import fft, ifft

def apply_filter(signal, cutoff_low=None, cutoff_high=None, frequency_hz=50):
    """
        Apply a frequency-based filter to a time-domain signal by filtering out specific frequencies.
        signal: the time-domain input signal (1D array or Series)
        cutoff_low: lower cutoff frequency for the filter (None means no low cutoff)
        cutoff_high: upper cutoff frequency for the filter (None means no high cutoff)
        frequency_hz: sampling frequency of the signal
    """
    # Ensure the signal is a numpy array and is contiguous
    signal = np.ascontiguousarray(signal)

    # Fourier Transform to frequency domain
    freqs = np.fft.fftfreq(len(signal), d=1/frequency_hz)
    fft_vals = fft(signal)

    # Create a mask for the desired frequencies
    mask = np.ones(len(freqs), dtype=bool)
    if cutoff_low is not None:
        mask = mask & (np.abs(freqs) >= cutoff_low)
    if cutoff_high is not None:
        mask = mask & (np.abs(freqs) <= cutoff_high)

    # Apply mask to zero out undesired frequencies
    fft_filtered = fft_vals * mask

    # Inverse FFT to convert back to time domain
    filtered_signal = np.real(ifft(fft_filtered))
    
    return filtered_signal
