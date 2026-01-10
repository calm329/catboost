import numpy as np
from pandas import DataFrame


from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator


from catboost import (
    CatBoostClassifier,
    CatBoostRegressor,
)


def test_sklearn_meta_algo():
    X_train = DataFrame(
        data=np.random.randint(0, 100, size=(100, 5)),
        columns=['feature{}'.format(i) for i in range(5)]
    )
    y_train = np.random.randint(0, 2, size=100)

    model = CatBoostClassifier()
    model.fit(X_train, y_train)

    cc_model = CalibratedClassifierCV(FrozenEstimator(model), method='isotonic')
    model = cc_model.fit(X_train, y_train)


def test_warm_start_classifier():
    X = DataFrame(data=np.random.randint(0, 100, size=(100, 5)), columns=[f'f{i}' for i in range(5)])
    y = np.random.randint(0, 2, size=100)

    model = CatBoostClassifier(iterations=10, warm_start=True, verbose=False)
    model.fit(X, y)
    assert model.tree_count_ == 10

    model.fit(X, y)
    assert model.tree_count_ == 20

    model.fit(X, y)
    assert model.tree_count_ == 30


def test_warm_start_regressor():
    X = DataFrame(data=np.random.randint(0, 100, size=(100, 5)), columns=[f'f{i}' for i in range(5)])
    y = np.random.rand(100)

    model = CatBoostRegressor(iterations=10, warm_start=True, verbose=False)
    model.fit(X, y)
    assert model.tree_count_ == 10

    model.fit(X, y)
    assert model.tree_count_ == 20


def test_warm_start_disabled():
    X = DataFrame(data=np.random.randint(0, 100, size=(100, 5)), columns=[f'f{i}' for i in range(5)])
    y = np.random.randint(0, 2, size=100)

    model = CatBoostClassifier(iterations=10, verbose=False)
    model.fit(X, y)
    assert model.tree_count_ == 10

    model.fit(X, y)
    assert model.tree_count_ == 10


def test_warm_start_with_init_model():
    X = DataFrame(data=np.random.randint(0, 100, size=(100, 5)), columns=[f'f{i}' for i in range(5)])
    y = np.random.randint(0, 2, size=100)

    init_model = CatBoostClassifier(iterations=5, verbose=False)
    init_model.fit(X, y)
    assert init_model.tree_count_ == 5

    model = CatBoostClassifier(iterations=10, warm_start=True, verbose=False)
    model.fit(X, y)
    assert model.tree_count_ == 10

    model.fit(X, y, init_model=init_model)
    assert model.tree_count_ == 15
