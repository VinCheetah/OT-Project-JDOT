import torch


class Model(torch.nn.Module):
	def __init__(self, n_epochs=100, device='cpu'):
		super(Model, self).__init__()
		self.model = torch.nn.Sequential(
			torch.nn.Flatten(start_dim=1),
			torch.nn.Linear(28 * 28, 128),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.5),
			torch.nn.Linear(128, 10),
			torch.nn.Softmax(dim=1)
		)
		self.n_epochs = n_epochs
		self.device = device
		self.criterion = torch.nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

	def fit(self, X, Y):
		# fit the model
		for epoch in range(self.num_epochs):
			self.model.train()
			X, Y = X.to(self.device), Y.to(self.device)
			Y = torch.nn.functional.one_hot(Y, num_classes=10).float()
			self.optimizer.zero_grad()
			output = self.model(X)
			loss = self.criterion(output, Y)
			loss.backward()
			self.optimizer.step()

	def predict(self, X):
		# predict the labels
		self.model.eval()
		X = X.to(self.device)
		output = self.model(X)
		_, predicted = torch.max(output)
		return predicted


def make_model(n_epochs=100, device='cpu'):
	return Model(n_epochs=n_epochs, device=device)