We generally follow these steps for preprocessing text data:
* **Tokenisation** : We split the text into individual sentences and then into individual words. We may also remove small words, numbers, unknown characters, punctuations, URLs and change everything to small case, depending on our application
* **Indexing** : Assign an index to each unique word. This is its one hot encoding vector. We generally start indexing from 2 and we reserve 0 for unknown words(words absent in our embedding layer) and 1 for padding
* **Constant Length Sequences** : We may truncate long sentences into short ones. Similarly for short sentences, we may pad the padding character. Pre-padding i.e padding before the sentence starts is preferred as RNNs and LSTMs may forget the actual sentence in case of Post-padding due to short memory. Very often, the data is divided into batches, and we try to keep sentences of similar length in one batch, allowing for better training

## Text Processing in PyTorch
### Includes
```
import torchtext.data as ttd
from torchtext.vocab import GloVe
```
### Declaring Fields
```
TEXT = ttd.Field(
  sequential = True, # the text will go into an RNN, so we make it into sequences
  batch_first = True,
  lower = False, # Don't convert to lower case
  tokenize = 'spacy', # A commonly used tokenizer. We can remove this line to use no special tokenizer
  pad_first = True # pre-padding
)
```
```
LABEL = ttd.Field(sequential = False, use_vocab = False, is_target = True)
```
### Loading the dataset
```
dataset = ttd.TabularDataset(
  path = 'myfile.csv',
  format = 'csv',
  skip_header = True,
  fields = [('data',TEXT),('label',LABEL)] # map the headers in the csv to Field objects in the code
)
```
### Train-Test split
```
train_dataset, test_dataset = dataset.split(0.8)
```
### Making a vocabulary
```
TEXT.build_vocab(train_dataset,) # notice the peculiar syntax
vocab = Text.vocab
```
### Making the generators
```
train_iter, test_iter = ttd.Iterator.splits(
  (train_dataset, test_dataset), 
  sort_key = lambda x : len(x.data), # The batches returned are sorted by length of sentence, so that sentences of similar length are in the same batch
  batch_size = (32,256), # We have a batch size of 32 for the train data and 256 for the test, since there is not back prop for the test data
  device = device
)
```
### Using the generators
```
for inputs, targets in train_iter :
  ...
```
