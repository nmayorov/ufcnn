import tensorflow as tf


def init_normal(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def conv(x, w, b, filter_length, dilation):
    padding = [[0, 0], [0, 0], [dilation * (filter_length - 1), 0], [0, 0]]
    x = tf.pad(x, padding)
    if dilation == 1:
        x = tf.nn.conv2d(x, w, [1, 1, 1, 1], padding='VALID')
    else:
        x = tf.nn.atrous_conv2d(x, w, dilation, padding='VALID')

    return x + b


def construct_ufcnn(n_inputs=1, n_outputs=1, n_levels=1, n_filters=10,
                    filter_length=5):
    """Construct a Undecimated Fully Convolutional Neural Network.

    The architecture replicates one from the paper [1]_. It is depicted below
    for 3 levels::

        input -- H1 ---------------------------- G1 -- C -- output
                     |                        |
                     -- H2 -------------- G2 --
                           |            |
                           -- H3 -- G3 --

    Here H and G are convolutional layers, each followed by ReLU
    transformation, C is the final convolutional layer. The outputs are
    concatenated at branch merges. All filter (except C) outputs `n_filters`
    signals, but because of concatenations filter G1 and G2 have to process
    2 * `n_filters` signals.

    A filter on level l implicitly contains 2**(l-1) zeros inserted between
    its values. It allows the network to progressively look farther into the
    past and learn dependencies on wide range of time scales.

    The important thing in time-series modeling is applying filters in a
    causal-way, i.e. convolutions must not include values after a current
    time moment. This is achieved by zero-padding from the left before
    applying the convolution.

    Implementation is done in tensorflow.

    Parameters
    ----------
    n_inputs : int, default 1
        Number of input time series.
    n_outputs : int, default 1
        Number of output time series.
    n_levels : int, default 1
        Number of levels in the network, see the picture above.
    n_filters : int, default 10
        Number of filters in each convolutional layers (except the last one).
    filter_length : int, default 5
        Length of filters.

    Returns
    -------
    x : tensorflow placeholder
        Placeholder representing input sequences. Use it to feed the input
        sequence into the network. The shape must be
        (batch_size, 1, n_stamps, `n_inputs`). Here n_stamps is the number
        of time stamps in the series. The second dimension has to be preserved
        because tensorflow doesn't fully support 1-dimensional data yet.
    y_hat : tensorflow placeholder
        Placeholder representing predicted output sequences. Use it to read-out
        networks predictions. The shape is
        (batch_size, 1, n_stamps, `n_outputs`).
    y : tensorflow placeholder
        Placeholder representing true output sequences. Use it to feed ground
        truth values to a loss operator during training of the network. For
        example, MSE loss can be defined as follows:
        ``loss = tf.reduce_mean(tf.square(y - y_hat))``. The shape must be
        the same as of `y_hat`.
    weights : list of tensorflow variables, length 2 * `n_levels` + 1
        List of convolution weights, the order is H, G, C.
    biases : list of tensorflow variables, length 2 * `n_levels` + 1
        List of convolution biases, the order is H, G, C.

    Notes
    -----
    Weights and biases will be initialized with truncated normal random
    variables with std of 0.1, you can reinitialize them using the returned
    `weights` and `biases` lists.

    References
    ----------
    .. [1] Roni Mittelman "Time-series modeling with undecimated fully
           convolutional neural networks", http://arxiv.org/abs/1508.00317
    """
    H_weights = []
    H_biases = []
    G_weights = []
    G_biases = []

    for level in range(n_levels):
        if level == 0:
            H_weights.append(
                init_normal([1, filter_length, n_inputs, n_filters]))
        else:
            H_weights.append(init_normal([1, filter_length,
                                          n_filters, n_filters]))
        H_biases.append(init_normal([n_filters]))
        if level == n_levels - 1:
            G_weights.append(init_normal([1, filter_length,
                                          n_filters, n_filters]))
        else:
            G_weights.append(init_normal([1, filter_length,
                                          2 * n_filters, n_filters]))

        G_biases.append(init_normal([n_filters]))

    x_in = tf.placeholder(tf.float32, shape=(None, 1, None, n_inputs))
    x = x_in
    level_outputs = []
    dilation = 1
    for w, b in zip(H_weights, H_biases):
        x = tf.nn.relu(conv(x, w, b, filter_length, dilation))
        level_outputs.append(x)
        dilation *= 2

    x_prev = None
    for x, w, b in zip(reversed(level_outputs),
                       reversed(G_weights),
                       reversed(G_biases)):
        if x_prev is not None:
            x = tf.concat(3, [x, x_prev])
        x = tf.nn.relu(conv(x, w, b, filter_length, dilation))
        x_prev = x
        dilation //= 2

    C_weights = init_normal([1, filter_length, n_filters, n_outputs])
    C_biases = init_normal([n_outputs])
    y_hat = conv(x, C_weights, C_biases, filter_length, 1)
    y = tf.placeholder(tf.float32, shape=(None, 1, None, n_outputs))

    weights = H_weights + G_weights + [C_weights]
    biases = H_biases + G_biases + [C_biases]

    return x_in, y_hat, y, weights, biases
