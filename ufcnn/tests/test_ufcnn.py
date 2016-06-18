import numpy as np
from numpy.testing import run_module_suite, assert_, assert_allclose
import tensorflow as tf

from ufcnn import construct_ufcnn, cross_entropy_loss, mse_loss, softmax
from ufcnn.datasets import generate_ar


def test_softmax():
    x = tf.placeholder(tf.float32, (3, 1, 2))
    sf = softmax(x)

    x_value = np.array([
        [[1, 2]],
        [[-1, 1]],
        [[0, 1]]
    ])
    e = np.exp(x_value)
    norm = np.sum(e, axis=-1)
    sf_true = e / norm[:, :, None]

    session = tf.Session()
    sf_computed = session.run(sf, feed_dict={x: x_value})
    session.close()

    # Accuracy in single precision.
    assert_allclose(sf_computed, sf_true)


def test_cross_entropy():
    x = tf.placeholder(tf.float32, (3, 1, 2))
    labels = tf.placeholder(tf.int32, (3, 1))
    ce = cross_entropy_loss(x, labels)

    x_value = np.array([
        [[1, 2]],
        [[-1, 1]],
        [[0, 1]]
    ])
    e = np.exp(x_value)
    norm = np.sum(e, axis=-1)
    sf_true = e / norm[:, :, None]

    labels_value = np.array([[0], [1], [1]])
    ce_true = -np.sum(np.log(sf_true[np.arange(3), :, labels_value.ravel()]))

    session = tf.Session()
    ce_computed = session.run(ce, feed_dict={x: x_value, labels: labels_value})
    session.close()

    assert_allclose(ce_computed, ce_true)


def test_reasonableness():
    # Run the net on a linear auto-regressive series and see if RMSE is
    # good after the training.

    X_train, Y_train = generate_ar(50, 400)
    X_test, Y_test = generate_ar(10, 400)

    for n_levels in [1, 2]:
        x, y_hat, *_ = construct_ufcnn(n_levels=n_levels)
        y = tf.placeholder(tf.float32, shape=(None, None, 1))

        loss = mse_loss(y_hat, y)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
        train_step = optimizer.minimize(loss)

        session = tf.Session()
        session.run(tf.initialize_all_variables())

        batch_size = 5
        n_batch = X_train.shape[0] // batch_size
        n_epochs = 20

        for epoch in range(n_epochs):
            for batch in range(n_batch):
                X_batch = X_train[batch * batch_size:(batch + 1) * batch_size]
                Y_batch = Y_train[batch * batch_size:(batch + 1) * batch_size]
                session.run(train_step, feed_dict={x: X_batch, y: Y_batch})

        mse = session.run(loss, feed_dict={x: X_test, y: Y_test})

        # Theoretically achievable RMSE is 0.1.
        assert_(mse**0.5 < 0.11)

        session.close()


if __name__ == '__main__':
    run_module_suite(argv=["", "--nologcapture"])
