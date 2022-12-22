import numpy as np


class spectral_derivative():

    def __init__(self, func, grid):
        self.func = func
        self.grid = grid

    @staticmethod
    def butterworth_filter(freqs, number_of_freqs, steepness):
        freqs_copy = np.copy(freqs)
        freqs_copy = np.abs(freqs_copy)
        freqs_copy.sort()
        butterworth_filter_multiplier = 1 / (1 + (freqs / freqs_copy[number_of_freqs - 1]) ** (2 * steepness))
        return freqs * butterworth_filter_multiplier

    def spectral_derivative_1d(self, func=None, grid=None, n=None, steepness=1):
        if isinstance(func, type(None)) and isinstance(grid, type(None)):
            func = self.func
            grid = self.grid
        func_projection = np.fft.rfft(func)
        if isinstance(n, type(None)):
            n = func_projection.size
        func_projection_copy = np.copy(func_projection)
        spacing_vector = np.reshape(grid, (1, grid.size))
        func_projection_filtered = func_projection_copy
        frequencies = np.fft.rfftfreq(spacing_vector.size, d=(spacing_vector[0][1] - spacing_vector[0][0]))
        frequencies_filtered = self.butterworth_filter(frequencies, n, steepness)
        return np.real(np.fft.irfft(1j * 2 * np.pi * frequencies_filtered * func_projection_filtered))

    def spectral_derivative_nd(self, n=None, steepness=1):
        if isinstance(n, int) or isinstance(n, type(None)):
            n = np.full(shape=len(self.grid), fill_value=n)
        all_dim_derivative = []
        inverter = lambda x: 1 if x == 0 else (x if x != 1 else 0)
        counter = 0
        for i in self.grid:
            all_dim_derivative.append(
                np.apply_along_axis(self.spectral_derivative_1d, inverter(counter), self.func, i, n[counter],
                                    steepness))
            counter += 1
        return all_dim_derivative
