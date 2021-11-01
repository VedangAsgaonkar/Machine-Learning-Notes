### Gradient Descent
[Learning Parameters, Part 1: Gradient Descent](https://towardsdatascience.com/learning-parameters-part-1-eb3e8bb9ffbb)

### Momentum
This optimizer helps to overcome regions of low slope [Learning Parameters, Part 2: Momentum-Based & Nesterov Accelerated Gradient Descent
](https://towardsdatascience.com/learning-parameters-part-2-a190bef2d12)

### Stochastic Gradient Descent
Stochastic Gradient Descent is different from Gradient Descent in the fact that Gradient Descent takes the whole data and finds the loss, while SGD just takes a random sample subset of the data. [Learning Parameters, Part 3: Stochastic & Mini-Batch Gradient Descent](https://towardsdatascience.com/learning-parameters-part-3-ee8558f65dd7)

### Choosing the learning rate
We can judge how good our learning rate is by seeing the graph of losses converging. We generally try powers of 10 like 0.1, 0.01, 0.001. [Learning Parameters, Part 4: Tips For Adjusting Learning Rate, Line Search](https://towardsdatascience.com/learning-parameters-part-4-6a18d1d3000b)

### Powerful Optimizers
Adagrad and RMSProp take smaller steps near convergence and also help to train sparse parameters. Adam combines this advantage with the speed of momentum and is more or less the default choice. [Learning Parameters, Part 5: AdaGrad, RMSProp, and Adam](https://towardsdatascience.com/learning-parameters-part-5-65a2f3583f7d)
