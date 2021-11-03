### Data representation
It is a convention in most ML libraries to represent time series data as N x T x D,
where T is the number of timesteps over which the data was measures, D is the number
of features that were measured in every timestep and N is the number of data samples (as 
all the samples are stacked together for vectorization). 
### Variable Length Sequences
If the sequences are variable length, we can convert to fixed length sequences by padding zeros and truncationg. As a compromise, we can have batches of different lengths, each batch has similar length of sequences. This is implemented in pytorch data loaders

### Simple RNN
```
class SimpleRNN(nn.Module):
  def __init__(self, n_input, n_hidden, n_rnnlayers, n_outputs):
    self.D = n_input # how large is the input for one sample at one time step
    self.M = n_hidden # how large in the hidden state vector
    self.L = n_rnnlayers # how many rnn layers to stack on top of one another. Note that this has nothing to do with the size of the time series
    self.K = n_outputs # final output size
    # batch_first = True so that we have a properly formatted output
    self.RNN = nn.RNN( input_size = self.D, hidden_size = self.M, num_layers = self.L, nonlinearity = 'relu', batch_first = True)
    self.fc = nn.Linear(self.M, self.K)
    
  def forward(self, X):
    # The rnn is not aware about the size of the sample N, and the length of the time series T beforehand
    
    h0 = torch.zeros(self.L, X.size(0), self.M).to(device) 
    # we initialize the hidden state vector, which is M x 1 for each rnn layer(self.L) and each sample (X.size(0))
    
    out, _ = self.rnn(X, h0)
    # note that the rnn automatically runs through the T timesteps
    # the first return variable is the hidden state vector for each time step in the last rnn layer. So it is N x T x M
    # the second return variable is the hidden state vector for each rnn layer at the last time step. So it is N x L x M
    
    # we are only going to use the last timestep of the last RNN layer here
    out = self.fc(out[:,-1,:]
    return out
```
### LSTM
An LSTM can be made in pytorch similar to an RNN
```
class LSTM(nn.Module):
  def __init__(self, n_input, n_hidden, n_rnnlayers, n_outputs):
    self.D = n_input # how large is the input for one sample at one time step
    self.M = n_hidden # how large in the hidden state vector
    self.L = n_rnnlayers # how many rnn layers to stack on top of one another. Note that this has nothing to do with the size of the time series
    self.K = n_outputs # final output size
    # batch_first = True so that we have a properly formatted output
    self.RNN = nn.LSTM( input_size = self.D, hidden_size = self.M, num_layers = self.L, nonlinearity = 'relu', batch_first = True)
    self.fc = nn.Linear(self.M, self.K)
    
  def forward(self, X):
    # The rnn is not aware about the size of the sample N, and the length of the time series T beforehand
    
    h0 = torch.zeros(self.L, X.size(0), self.M).to(device) 
    c0 = torch.zeros(self.L, X.size(0), self.M).to(device) 
    # we initialize the hidden state vector, which is M x 1 for each rnn layer(self.L) and each sample (X.size(0))
    
    out, _ = self.rnn(X, (h0,c0))
    # note that the rnn automatically runs through the T timesteps
    # the first return variable is the hidden state vector for each time step in the last rnn layer. So it is N x T x M
    # the second return variable is the hidden state vector for each rnn layer at the last time step. So it is N x L x M
    
    # we are only going to use the last timestep of the last RNN layer here
    out = self.fc(out[:,-1,:]
    return out
```
    
