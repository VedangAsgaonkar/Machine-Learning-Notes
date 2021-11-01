We can perform test train split as follows
```
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, targets, test_size=0.2)
N, D = X_train.shape
```
