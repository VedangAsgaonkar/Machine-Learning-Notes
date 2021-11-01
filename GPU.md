In Pytorch, it is easy to load the model on a GPU if available. This can be done with
```
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
```
If this is done, then the inputs and targets need to be explicitly sent to the device using
```
inputs, targets = inputs.to(device), targets.to(device)
```
