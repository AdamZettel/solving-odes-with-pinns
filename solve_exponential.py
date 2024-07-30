import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class ODEModel(nn.Module):
    def __init__(self):
        super(ODEModel, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        self.relu = nn.ReLU()

    def forward(self, t):
        x = self.relu(self.fc1(t))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def compute_loss(model, t, k, y0):
    # Forward pass
    y_pred = model(t)
    
    # Compute the derivative of y with respect to t
    t.requires_grad_(True)
    y_pred = model(t)
    dy_dt = torch.autograd.grad(y_pred, t, torch.ones_like(t), create_graph=True)[0]
    
    # The differential equation is dy/dt = ky
    ode_loss = torch.mean((dy_dt - k * y_pred) ** 2)
    
    # Initial condition loss
    y0_pred = model(torch.tensor([[0.0]]))
    ic_loss = torch.mean((y0_pred - y0) ** 2)
    
    # Total loss
    total_loss = ode_loss + ic_loss
    return total_loss

# Hyperparameters
k = 0.5  # Growth rate
y0 = 1.0  # Initial condition
learning_rate = 0.01
num_epochs = 1000

# Create the model, loss function, and optimizer
model = ODEModel()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Create a tensor for time values
t = torch.linspace(0, 1, 100).view(-1, 1)

# Training loop
for epoch in range(num_epochs):
    model.train()
    
    optimizer.zero_grad()
    
    # Compute the loss
    loss = compute_loss(model, t, k, y0)
    
    # Backward pass and optimization
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item()}')

# Testing the model
model.eval()
t_test = torch.linspace(0, 1, 100).view(-1, 1)
y_test = model(t_test)

# Compute the analytical solution
t_np = t_test.detach().numpy()
y_analytical = y0 * np.exp(k * t_np)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t_test.detach().numpy(), y_test.detach().numpy(), label='Neural Network Prediction', linestyle='--')
plt.plot(t_np, y_analytical, label='Analytical Solution', linestyle='-')
plt.title('Comparison of Neural Network Solution with Analytical Solution')
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.legend()
plt.show()
