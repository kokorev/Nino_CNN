#!/usr/bin/env python
# coding=utf-8
"""

"""


if __name__ == '__main__':
    import numpy as np
    from reader import Reader
    from model import get_model

    params = dict(
        test_month=5 * 12,
        lookback=4,
        epochs_initial=10,
        epochs_per_opa=15,
        seed=False,
        noise=False,
        n_forward=6
    )
    loss_history = []

    pp = Reader()
    model = get_model(params['noise'], lookback=params['lookback'])

    # Train model
    # Phase 0
    print('Initial training')
    X, y = pp.get_xy(0, params['n_forward'], be=True)
    history_callback = model.fit(X, y, epochs=params['epochs_initial'])
    loss_history.append(history_callback.history["loss"])

    # Phase 1
    for opa in range(5):
        print('Learning from opa{}'.format(opa))
        X, y = pp.get_xy(opa, params['n_forward'])
        X_train, X_test, y_train, y_test = pp.train_test_split_tail(X, y, params['test_month'])
        history_callback = model.fit(X_train, y_train, epochs=params['epochs_per_opa'])
        loss_history.append(history_callback.history["loss"])

    # Evaluate
    y_pred = model.predict(X_test)
    corr = np.corrcoef(y_test, y_pred[:, 0])[1, 0]
    print(' Predicted vs True Correlation - ', corr)

