The basic loop for PyTorch
```
model = nn.Linear(1,1) # specify model
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.07)
n_epochs = 100
train_losses = np.zeros(n_epochs)
test_losses = np.zeros(n_epochs)
for i in range(n_epochs):
  optimizer.zero_grad()
  outputs = model(inputs)
  loss = criterion(outputs,targets)
  loss.backward()
  optimizer.step()
  test_outputs = model(test_inputs)
  test_loss = criterion(test_outputs, test_targets)
  train_losses[it] = loss.item()
  test_losses[it] = test_loss.item()
  print(f'Epoch {it+1}/{n_epochs}, Train Loss: {loss.item():.4f}, Test Loss: {loss_test.item():.4f}')
plt.plot(train_losses, label='train loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()
 ```
