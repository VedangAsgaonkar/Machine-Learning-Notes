Word embeddings are a way to capture similarity between words in their vector
representations. [Introduction to Word Embedding and Word2Vec](https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa).
There are two main algorithms used to make word embeddings :
* [Word2Vec — Skip-gram and CBOW](https://towardsdatascience.com/nlp-101-word2vec-skip-gram-and-cbow-93512ee24314)
* [Negative Sampling and GloVe](https://towardsdatascience.com/nlp-101-negative-sampling-and-glove-936c88f3bc68)

### Embeddings in PyTorch
* **Pre-trained**:[How to use Pre-trained Word Embeddings in PyTorch](https://medium.com/@martinpella/how-to-use-pre-trained-word-embeddings-in-pytorch-71ca59249f76)
* **Trainable**: We can simply use ```nn.Embedding(vocab_size, embedding_size)``` in the model. In the default layer that PyTorch provides, the weights are randomly initialized by sampling from a gaussian distribution. We can change the initialisation like
```
self.emb.weight.data = nn.Parameter(torch.Tensor(np.random.randn(self.N, self.D) * 0.01))
```
The embedding layer in pytorch directly takes in an index as input rather than a one hot vector, saving space
