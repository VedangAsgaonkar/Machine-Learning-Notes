To deal with data in batches(so as to not overload the memory), we can use data loaders provided by pytorch. These are 
generators, so we can iterate over them
```
batch_size = 128
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
```
Now the training loop will look like
```
for it in range(n_epochs):
  for inputs, targets in train_loader:
    ...
  for inputs, targets in test_loader:
    ...
```
