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

	def fit(self, X, y):
		# fit the model
		y = torch.nn.functional.one_hot(y, num_classes=10).float()
		for epoch in range(self.n_epochs):
			self.model.train()
			X, y = X.to(self.device), y.to(self.device)
			self.optimizer.zero_grad()
			output = self.model(X)
			loss = self.criterion(output, y)
			loss.backward()
			self.optimizer.step()

	def predict(self, X):
		# predict the labels
		self.model.eval()
		X = X.to(self.device)
		output = self.model(X)
		_, predicted = torch.max(output, dim=1)
		return predicted


def make_model(n_epochs=100, device='cpu'):
	return Model(n_epochs=n_epochs, device=device)