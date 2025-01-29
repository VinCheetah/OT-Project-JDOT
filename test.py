import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define the CNN model
class MNIST_CNN(nn.Module):
	def __init__(self):
		super(MNIST_CNN, self).__init__()
		self.model = torch.nn.Sequential(
			torch.nn.Flatten(start_dim=1),
			torch.nn.Linear(28 * 28, 128),
			torch.nn.ReLU(),
			torch.nn.Dropout(0.5),
			torch.nn.Linear(128, 10),
			torch.nn.Softmax(dim=1)
		)
	def forward(self, x):
		return self.model(x)

# Load MNIST dataset
transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset_source = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset_source = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_dataset_target = torchvision.datasets.USPS(root='./data', train=True, download=True, transform=transform)
test_dataset_target = torchvision.datasets.USPS(root='./data', train=False, download=True, transform=transform)

train_loader_source = DataLoader(train_dataset_source, batch_size=64, shuffle=True)
test_loader_source = DataLoader(test_dataset_source, batch_size=1000, shuffle=False)

train_loader_target = DataLoader(train_dataset_target, batch_size=64, shuffle=True)
test_loader_target = DataLoader(test_dataset_target, batch_size=1000, shuffle=False)


print("Data loaded")
# print the number of samples in the training set and test set
print(f"Number of samples in the training source set: {len(train_dataset_source)}")
print(f"Number of samples in the test source set: {len(test_dataset_source)}")
print(f"Number of samples in the training target set: {len(train_dataset_target)}")
print(f"Number of samples in the test target set: {len(test_dataset_target)}")

# Initialize the model, loss function, and optimizer
device = torch.device("mps")
model = MNIST_CNN().to(device)
# criterion = nn.CrossEntropyLoss()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
	model.train()
	running_loss = 0.0
	for images, labels in train_loader_source:
		images, labels = images.to(device), labels.to(device)
		# use one hot encoding for the labels
		labels = torch.nn.functional.one_hot(labels, num_classes=10).float()
		
		optimizer.zero_grad()
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		
		running_loss += loss.item()
	print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader_source):.4f}")

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
	for images, labels in test_loader_source:
		images, labels = images.to(device), labels.to(device)
		outputs = model(images)
		_, predicted = torch.max(outputs, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")

# correct = 0
# total = 0
# with torch.no_grad():
# 	for images, labels in test_loader_target:
# 		images = torch.nn.functional.pad(images, (6, 6, 6, 6), mode='constant', value=-0.4242)
# 		images, labels = images.to(device), labels.to(device)
# 		outputs = model(images)
# 		_, predicted = torch.max(outputs, 1)
# 		total += labels.size(0)
# 		correct += (predicted == labels).sum().item()

# print(f"Test Accuracy: {100 * correct / total:.2f}%")


# Save the trained model
# torch.save(model.state_dict(), "mnist_cnn.pth")
# print("Model saved to mnist_cnn.pth")


# self.model = torch.nn.Sequential(
# 	torch.nn.Conv2d(1, 32, 3, padding='same'),
# 	torch.nn.ReLU(),
# 	# torch.nn.Conv2d(32, 64, 5, padding='same'),
# 	# torch.nn.ReLU(),
# 	torch.nn.Flatten(start_dim=1),
# 	torch.nn.Linear(32 * 28 * 28, 128),
# 	torch.nn.ReLU(),
# 	torch.nn.Dropout(0.5),
# 	torch.nn.Linear(128, 10),
# 	torch.nn.Softmax(dim=1)
# )