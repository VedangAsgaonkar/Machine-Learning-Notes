Modules in PyTorch are built from inheriting from the ```nn.Module``` class.
They can be stacked together to give bigger modules. A model is also a module.
Eg:
```
model = nn.Sequential(
  nn.Linear(784,128),
  nn.ReLU(),
  nn.Linear(128,1)
)
```
can also be made as
```
class ANN(nn.Module):
  def __init__(self):
    super(ANN, self).__init__()
    self.layer1 = nn.Linear(784, 128)
    self.layer2 = nn.ReLU()
    self.layer3 = nn.Linear(128,1)
  def forward(self,x):
    x = self.layer1(x)
    x = self.layer2(x)   
    x = self.layer3(x)
    return x
 ```
 ```
 model = ANN()
 ```
