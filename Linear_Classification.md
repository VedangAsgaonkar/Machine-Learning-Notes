In simple binary classification, the typical model is Sequential like
```
model = nn.Sequential(
          nn.Linear(D,1),
          nn.Sigmoid()
         )
```
The typical loss is ```nn.BCELoss()``` and optimizer is ```torch.optim.Adam(model.parameters())```
Also, we use accuracy and not loss as a metric for testing at the end. By accuracy we mean the fraction of data that was correctly classified. 
So we do
```
with model.no_grad():
  p_test = model(X_test)
  p_test = np.round(p_test.numpy())
  test_acc = np.mean(p_test == y_test.numpy())
```
For a multiclass classification, we use categorical cross entropy
