"""Tests whether the emulator with homGP gives the same result as just using homGP"""

import sys

sys.path.append("./")
import numpy as np
from hetgpy import homGP
from PUQ.surrogate import emulator

# generate some test data
rng = np.random.default_rng(123)

X = np.linspace(0, 2 * np.pi, 20).reshape(-1, 1)
# add some replicates
reps = rng.choice(len(X), size=100, replace=True)
X = X[reps,]
Ytrue = np.sin(X)
noise = 0.5 * rng.normal(size=(len(Ytrue), 1))

Y = Ytrue + noise

# prediction grid (interpolation)
Xgrid = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)


def test_fit():

    reference_model = homGP()
    reference_model.mle(X, Y.flatten())

    test_model = emulator(x=X, theta=np.array([0]), f=Y, method="homGP")
    test_model.fit()

    assert np.allclose(test_model._info["ll"], reference_model["ll"])
    assert np.allclose(test_model._info["theta"], reference_model["theta"])
    assert np.allclose(test_model._info["g"], reference_model["g"])
    assert np.allclose(test_model._info["beta0"], reference_model["beta0"])


def test_predict():
    # refits models and then tests predictions

    reference_model = homGP()
    reference_model.mle(X, Y.flatten())

    test_model = emulator(x=X, theta=np.array([0]), f=Y, method="homGP")
    test_model.fit()

    # test predictions
    preds = reference_model.predict(Xgrid, xprime=Xgrid)

    test_preds = test_model.predict(x=Xgrid, thetaprime=Xgrid)
    assert np.allclose(test_preds._info["mean"], preds["mean"])
    assert np.allclose(test_preds._info["var"], preds["sd2"])
    assert np.allclose(test_preds._info["nugs"], preds["nugs"])
    assert np.allclose(test_preds._info["covmat"], preds["cov"])


def test_predict_nugs_only():

    reference_model = homGP()
    reference_model.mle(X, Y.flatten())

    test_model = emulator(x=X, theta=np.array([0]), f=Y, method="homGP")
    test_model.fit()

    # test predictions
    preds = reference_model.predict(Xgrid, xprime=Xgrid, nugs_only=True)

    test_preds = test_model.predict(
        x=Xgrid, thetaprime=Xgrid, args=dict(nugs_only=True)
    )
    for key in ["mean", "var", "covmat"]:
        assert test_preds._info.get(key) is None
    assert np.allclose(test_preds._info["nugs"], preds["nugs"])


def test_matern():
    # test a different covariance type and define some input args
    COVTYPE = "Matern5_2"
    MAXIT = 50
    SETTINGS = {"return.Ki": False, "factr": 1e5}

    reference_model = homGP()
    reference_model.mle(X, Y.flatten(), covtype=COVTYPE, maxit=MAXIT, settings=SETTINGS)

    test_model = emulator(
        x=X,
        theta=np.array([0]),
        f=Y,
        method="homGP",
        args={"covtype": COVTYPE, "maxit": MAXIT, "settings": SETTINGS},
    )
    test_model.fit()

    assert np.allclose(test_model._info["ll"], reference_model["ll"])
    assert np.allclose(test_model._info["theta"], reference_model["theta"])
    assert np.allclose(test_model._info["g"], reference_model["g"])
    assert np.allclose(test_model._info["beta0"], reference_model["beta0"])


def test_homGP_update_kriging_believer():
    reference_model = homGP()
    reference_model.mle(X, Y.flatten())

    Xnew = X.mean().reshape(-1, 1)
    Ypred = reference_model.predict(Xnew)["mean"]
    reference_model.update(Xnew, Ypred, maxit=0)

    test_model = emulator(x=X, theta=np.array([0]), f=Y, method="homGP")
    test_model.fit()
    test_model.update(Xnew)

    assert np.allclose(test_model._info["ll"], reference_model["ll"])
    assert np.allclose(test_model._info["theta"], reference_model["theta"])
    assert np.allclose(test_model._info["g"], reference_model["g"])
    assert np.allclose(test_model._info["beta0"], reference_model["beta0"])


def test_homGP_update():
    reference_model = homGP()
    reference_model.mle(X, Y.flatten())

    Xnew = X.mean().reshape(-1, 1)
    Ypred = reference_model.predict(Xnew)["mean"]
    reference_model.update(Xnew, Ypred)

    test_model = emulator(x=X, theta=np.array([0]), f=Y, method="homGP")
    test_model.fit()
    test_model.update(x=Xnew, Y=Ypred)

    assert np.allclose(test_model._info["ll"], reference_model["ll"])
    assert np.allclose(test_model._info["theta"], reference_model["theta"])
    assert np.allclose(test_model._info["g"], reference_model["g"])
    assert np.allclose(test_model._info["beta0"], reference_model["beta0"])


if __name__ == "__main__":
    test_homGP_update()
