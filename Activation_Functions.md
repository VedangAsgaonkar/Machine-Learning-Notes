We have the following activation functions generally in use:
* Softmax : to return normalised probabilities
* tanh : Advantage : restricts values from becoming too large. Disadvantage : Slow training when values of input are large, vanishing gradient problem. [The Vanishing Gradient Problem](https://towardsdatascience.com/the-vanishing-gradient-problem-69bf08b15484)
* ReLU (Most common *️⃣) : Advantage : overcomes all problems of tanh. Disadvantage : No training in negative inputs
* Leeky ReLU : Advantage : overcomes issues of ReLU
