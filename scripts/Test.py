import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt



def testfunction(x):
    
    return np.sin(x)*np.exp(-x)


    
# Generate training data
x_train = np.linspace(-2 * np.pi, 2 * np.pi, 1000)    
y_train = testfunction(x_train)


# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train, dtype=torch.float32).view(-1, 1)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

# Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Initialize model, loss function, and optimizer
model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 5000
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(x_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 500 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Test and plot results
x_test = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).view(-1, 1)
y_pred = model(x_test_tensor).detach().numpy()

plt.plot(x_test, testfunction(x_test), label='True func.')
plt.plot(x_test, y_pred, label='MLP Approximation', linestyle='dashed')
plt.legend()
plt.show()
