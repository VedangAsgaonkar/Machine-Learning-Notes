### PyTorch
The PyTorch model is 
```model = nn.Linear(inputDim, outputDim)```
To print out weights and biases use
```
w = model.weight.data.numpy()
b = model.bias.data.numpy()
print(w,b)
```
