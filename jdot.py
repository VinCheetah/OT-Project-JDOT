import numpy as np
from scipy.spatial.distance import cdist
import ot
import torch

def jdot_nn_l2(get_model, X_s, y_s, X_t, y_t=[], reset_model=True, numIterBCD=10, alpha=1, method="emd", reg=0.1, n_epochs=100):
	"""
	JDOT with neural network and l2 loss

	Args:
		get_model (Callable): Function that returns a new model compiled with l2 loss
		X_s (list): Data from source domain
		y_s (list): Labels from source domain
		X_t (list): Data from target domain
		y_t (list, optional): Labels from target domain. Is used just to measure the performances fo the method along iterations. Defaults to [].
		reset_model (bool, optional): Boolean to reset the model at each iteration. Defaults to True.
		numIterBCD (int, optional): Number of Iterations for BCD. Defaults to 10.
		alpha (int, optional): Ponderation between ground cost + function cost. Defaults to 1.
		method (str, optional): Choice of algorithm for transport computation. Methods avalaible are: "emd", "sinkhorn". Defaults to "emd".
		reg (int, optional): Parameter for sinkhorn. Defaults to .1.
		n_epochs (int, optional): Number of epochs for training the model. Defaults to 100.

	Returns:
		model, results: Returns the model and the results
	"""

	# Initializations
	n_s = X_s.shape[0]
	n_t = X_t.shape[0]
 
	# Initialize uniform weights for transport
	w_unif_s = np.ones((n_s,)) / n_s
	w_unif_t = np.ones((n_t,)) / n_t

	# original loss
	C0 = cdist(X_s, X_t, metric="sqeuclidean")
	C0 /= np.max(C0)

	# classifier
	model = get_model(n_epochs)

	accuracy = []
	fcosts = []
	totalcosts = []

	# Init initial model
	model.fit(X_s, y_s)
	y_t_pred = model.predict(X_t)

	C = alpha * C0 + cdist(y_s, y_t_pred, metric="sqeuclidean")

	if len(y_t) > 0:
		accuracy.append((torch.argmax(y_t, axis=1) == torch.argmax(y_t_pred, axis=1)).float().mean())


	for num_iter in range(numIterBCD):
		if method == "sinkhorn":
			gamma = ot.sinkhorn(w_unif_s, w_unif_t, C, reg)
		elif method == "emd":
			gamma = ot.emd(w_unif_s, w_unif_t, C)
		else:
			raise ValueError("Method not implemented")

		y_st = torch.tensor(n_t * gamma.T.dot(y_s)).float()
  
		if reset_model:
			model = get_model(n_epochs)
		model.fit(X_t, y_st)
		y_t_pred = model.predict(X_t)

		if num_iter > 1:
			fcost = cdist(y_s, y_t_pred, metric="sqeuclidean")
			fcosts.append(np.sum(gamma * fcost))
			totalcosts.append(np.sum(gamma * (alpha * C0 + fcost)))

		if len(y_t) > 0:
			accuracy.append((torch.argmax(y_t, axis=1) == torch.argmax(y_t_pred, axis=1)).float().mean())

	results = {
		"ypred0": y_t_pred,
		"ypred": np.argmax(y_t_pred, 1) + 1,
		"clf": model,
		"fcost": fcosts,
		"totalcost": totalcosts,
	}
	if len(y_t):
		results["accuracy"] = accuracy
	return model, results
