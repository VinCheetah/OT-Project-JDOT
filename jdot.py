import numpy as np
from scipy.spatial.distance import cdist
import ot


def jdot_nn_l2(get_model, X, Y, Xtest, ytest=[], fit_params={}, reset_model=True, numIterBCD=10, alpha=1, method="emd", reg=1):
    """
    JDOT with neural network and l2 loss

    Args:
        get_model (Callable): Function that returns a new model compiled with l2 loss
        X (list): Data from source domain
        Y (list): Labels from source domain
        Xtest (list): Data from target domain
        ytest (list, optional): Labels from target domain. Is used just to measure the performances fo the method along iterations. Defaults to [].
        fit_params (dict, optional): _description_. Defaults to {}.
        reset_model (bool, optional): Boolean to reset the model at each iteration. Defaults to True.
        numIterBCD (int, optional): Number of Iterations for BCD. Defaults to 10.
        alpha (int, optional): Ponderation between ground cost + function cost. Defaults to 1.
        method (str, optional): Choice of algorithm for transport computation. Methods avalaible are: "emd", "sinkhorn". Defaults to "emd".
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

    # Init initial model
    model.fit(X, Y, **fit_params)
    ypred = model.predict(Xtest)

    C = alpha * C0 + cdist(Y, ypred, metric="sqeuclidean")

    # do it only if the final labels were given
    if len(ytest):
        ydec = np.argmax(ypred, 1) + 1
        TBR.append(np.mean(ytest == ydec))

    for num_iter in range(numIterBCD):
        match method:
            case "sinkhorn":
                G = ot.sinkhorn(wa, wb, C, reg)
            case "emd":
                G = ot.emd(wa, wb, C)
            case _:
                raise ValueError("Method not implemented")
            
        Yst = ntest * G.T.dot(Y)
        if reset_model:
            model = get_model()

        model.fit(Xtest, Yst, **fit_params)
        ypred = model.predict(Xtest)

        # function cost
        fcost = cdist(Y, ypred, metric="sqeuclidean")
        C = alpha * C0 + fcost

        ydec = np.argmax(ypred, 1) + 1
        if num_iter > 1:
            sav_fcost.append(np.sum(G * fcost))
            sav_totalcost.append(np.sum(G * (alpha * C0 + fcost)))

        if len(ytest):
            TBR.append(np.mean((ytest - ypred) ** 2))

    results = {
        "ypred0": ypred,
        "ypred": np.argmax(ypred, 1) + 1,
        "clf": model,
        "fcost": sav_fcost,
        "totalcost": sav_totalcost,
    }
    if len(ytest):
        results["mse"] = TBR
    return model, results
