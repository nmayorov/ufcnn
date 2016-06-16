from numpy.testing import run_module_suite, assert_

import tensorflow as tf
from ufcnn import construct_ufcnn
from ufcnn.datasets import generate_ar


def test_reasonableness():
    # Run the net on a linear auto-regressive series and see if RMSE is
    # good after the training.

    X_train, Y_train = generate_ar(50, 400)
    X_test, Y_test = generate_ar(10, 400)

    for n_levels in [1, 2]:
        x, y_hat, y, *_ = construct_ufcnn(n_levels=n_levels)

        loss = tf.reduce_mean(tf.square(y_hat - y))
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01)
        train_step = optimizer.minimize(loss)

        sess = tf.Session()
        sess.run(tf.initialize_all_variables())

        batch_size = 5
        n_batch = X_train.shape[0] // batch_size
        n_epochs = 20

        for epoch in range(n_epochs):
            for batch in range(n_batch):
                X_batch = X_train[batch * batch_size:(batch + 1) * batch_size]
                Y_batch = Y_train[batch * batch_size:(batch + 1) * batch_size]
                sess.run(train_step, feed_dict={x: X_batch, y: Y_batch})

        mse = sess.run(loss, feed_dict={x: X_test, y: Y_test})

        # Theoretically achievable RMSE is 0.1.
        assert_(mse**0.5 < 0.11)

        sess.close()


if __name__ == '__main__':
    run_module_suite(argv=["", "--nologcapture"])
