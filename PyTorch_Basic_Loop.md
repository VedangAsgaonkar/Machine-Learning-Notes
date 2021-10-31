The basic loop for PyTorch
```
model = nn.Linear(1,1) # specify model
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.07)
n_epochs = 100
losses = []
for i in range(n_epochs):
  optimizer.zero_grad()
  outputs = model(inputs)
  loss = criterion(outputs,targets)
  losses.append(loss)
  loss.backward()
  optimizer.step()
  printf(print(f'Epoch {it+1}/{n_epochs}, Loss: {loss.item():.4f}'))
 plt.plot(losses)
 ```
