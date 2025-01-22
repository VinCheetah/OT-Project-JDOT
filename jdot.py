import numpy as np
from scipy.spatial.distance import cdist
import ot

# X: source domain
# y: source labeks
# Xtest: target domain
# ytest is optionnal, just to measure performances of the method along iterations
# gamma: RBF kernel param (default=1)
# numIterBCD: number of Iterations for BCD (default=10)
# alpha: ponderation between ground cost + function cost
# method: choice of algorithm for transport computation (default: emd)


def jdot_nn_l2(get_model, X, Y, Xtest, ytest=[], fit_params={}, reset_model=True, numIterBCD=10, alpha=1, method="emd", reg=1, nb_epoch=100, batch_size=10):
    """
    JDOT with neural network and l2 loss

    Args:
        get_model (Callable): Function that returns a new model compiled with l2 loss
        X (list): Data from source domain
        Y (list): Labels from source domain
        Xtest (list): Data from target domain
        ytest (list, optional): Labels from target domain. Defaults to [].
        fit_params (dict, optional): _description_. Defaults to {}.
        reset_model (bool, optional): Boolean to reset the model at each iteration. Defaults to True.
        numIterBCD (int, optional): Number of Iterations for BCD. Defaults to 10.
        alpha (int, optional): Ponderation between ground cost + function cost. Defaults to 1.
        method (str, optional): Choice of algorithm for transport computation. Defaults to "emd".
        reg (int, optional): Parameter for sinkhorn. Defaults to 1.

    Returns:
        model, results: Returns the model and the results
    """

    # Initializations
    n = X.shape[0]
    ntest = Xtest.shape[0]
    wa = np.ones((n,)) / n
    wb = np.ones((ntest,)) / ntest

    # original loss
    C0 = cdist(X, Xtest, metric="sqeuclidean")
    C0 /= np.max(C0)

    # classifier
    model = get_model()

    TBR = []
    sav_fcost = []
    sav_totalcost = []

    results = {}

    # Init initial g(.)
    model.fit(X, Y, **fit_params)
    ypred = model.predict(Xtest)

    C = alpha * C0 + cdist(Y, ypred, metric="sqeuclidean")

    # do it only if the final labels were given
    if len(ytest):
        ydec = np.argmax(ypred, 1) + 1
        TBR1 = np.mean(ytest == ydec)
        TBR.append(TBR1)

    k = 0
    while k < numIterBCD:  # and not changeLabels:
        k = k + 1
        if method == "sinkhorn":
            G = ot.sinkhorn(wa, wb, C, reg)
        if method == "emd":
            G = ot.emd(wa, wb, C)

        Yst = ntest * G.T.dot(Y)
        if reset_model:
            model = get_model()

        model.fit(Xtest, Yst, **fit_params)
        ypred = model.predict(Xtest)

        # function cost
        fcost = cdist(Y, ypred, metric="sqeuclidean")
        C = alpha * C0 + fcost

        ydec_tmp = np.argmax(ypred, 1) + 1
        if k > 1:
            sav_fcost.append(np.sum(G * fcost))
            sav_totalcost.append(np.sum(G * (alpha * C0 + fcost)))

        ydec = ydec_tmp
        if len(ytest):
            TBR1 = np.mean((ytest - ypred) ** 2)
            TBR.append(TBR1)

    results["ypred0"] = ypred
    results["ypred"] = np.argmax(ypred, 1) + 1
    if len(ytest):
        results["mse"] = TBR
    results["clf"] = model
    results["fcost"] = sav_fcost
    results["totalcost"] = sav_totalcost
    return model, results
