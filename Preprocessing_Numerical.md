### Normalizing
We should always normalize our features and labels. [Why Data Normalization is necessary for Machine Learning models](https://medium.com/@urvashilluniya/why-data-normalization-is-necessary-for-machine-learning-models-681b65a05029).
This can be done by:
```
X_normalized = (X-X.mean())/X.std()
```
We generally normalize using scikit-learn
```
from sklearn.preprocessing import StandardScaler
scaler = StandardScalar
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test) # fit the parameters according to the train data
```
where X is a numpy vector. Remember to convert back to original scale in case you are interested in values of particular weights/biases like in linear regression.

### Visualising 
Scatter plots can be used to visualise relation between variables using ```plt.scatter(X,Y)```

### Transforming
If we are looking at an exponential relation, a transformation like ```np.log()``` may help. Similiarly for quadratics, ```np.sqrt()``` is useful

### Datatype
Convert to float using numpy. For vectors(not matrices), we should use ```Y.reshape(1,-1)``` so that it doesn't have a partial shape
```
X = X.astype(np.float32)
```
Then convert to Torch Tensor
```
X = torch.from_numpy(X)
```
