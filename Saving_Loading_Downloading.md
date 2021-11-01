Consider the model trained from
```
model = nn.Sequential(nn.Linear(D,1), nn.Sigmoid())
```
After training, we can view model as
```
model.state_dict()
```
To save the model use
```
torch.save(model.state_dict(), 'mymodel.pt')
```
To load a saved model
```
model2 = nn.Sequential(nn.Linear(D,1), nn.Sigmoid)
model2.load_state_dict(torch.load('mymodel.pt'))
```
To Download a model
```
from google.colab import files
files.download('mymodel.pt')
```
