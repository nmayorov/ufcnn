"""Example data sets."""
import numpy as np


def generate_tracking(n_series, n_stamps, speed=0.2,
                      dynamics_noise=0.005, measurement_noise=0.005,
                      random_state=0):
    """Generate data from the tracking problem from [1]_.

    The task is to estimate the position of a target moving with a constant
    speed in a 2-dimensional square box, bouncing from its bounds, using the
    measurements of its polar angle (bearing). The box is centered at (0, 0)
    and its side length is 20.

    Parameters
    ----------
    n_series : int
        Number of time series to generate.
    n_stamps : int
        Number of stamps in each time series.
    speed : float, default 0.1
        Step size of the target per time stamp.
    dynamics_noise : float, default 0.005
        Standard deviation of noise to add to the target position.
    measurement_noise : float, default 0.05
        Standard deviation of noise to add to the bearing measurements.
    random_state : int, default 0
        Seed to use in the random generator.

    Returns
    -------
    X : ndarray, shape (n_series, 1, n_stamps, 1)
        Input series.
    Y : ndarray, shape (n_series, 1, n_stamps, 2)
        Output series of x and y coordinates.

    References
    ----------
    .. [1] Roni Mittelman "Time-series modeling with undecimated fully
           convolutional neural networks", http://arxiv.org/abs/1508.00317.
    """
    rng = np.random.RandomState(random_state)
    angle = rng.uniform(-np.pi, np.pi, n_series)
    velocity = speed * np.vstack((np.sin(angle), np.cos(angle))).T
    position = np.arange(n_stamps)[None, :, None] * velocity[:, None, :]
    position += dynamics_noise * rng.randn(*position.shape)

    D = 10
    t = np.remainder(position + D, 4 * D)
    position = -D + np.minimum(t, 4 * D - t)
    bearing = np.arctan2(position[:, :, 1], position[:, :, 0])
    bearing += measurement_noise * rng.randn(*bearing.shape)

    X = bearing.reshape((n_series, 1, n_stamps, 1))
    Y = position.reshape((n_series, 1, n_stamps, 2))

    return X, Y
