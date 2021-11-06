### Preprocessing
Empirical evidence shows that GANs work better when pixel values are between -1 and 1. So we
transform our sample images into this format using
```
transform = torchvision.transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize(mean = (0.5,), std=(0.5,))
])
```
We also add a ```nn.Tanh()``` at the end of the Generator to achieve this for the
fake images. To render the generated images, we define a function
```
def scale_image(img):
  out = (img+1)/2
  return out
```
to bring pixels back to (0,1) range

### Discriminator
The discriminator is generally a simple feedforward NN with 2-3 layers, each with a decreasing
number of Neurons. Its job is that of a linear classifier

### Generator
The generator has a ```latent_dim``` hyperparameter, the size of the multivariate gaussian
it would sample from.
```
noise = torch.randn(inputs.size(0), latent_dim).to(device)
```
The generator is more complex than the discriminator. It is feedforward NN, which first increases the
number of neurons per layer and at the end reduces to the size of the image. It also has tanh at the end.
The generator may also use BatchNorm1d layers in between.

### Training
There is a common loss function for both models at the end of the discriminator, generally cross entropy.
Both models have seperate optimizers which are associated with their own parameters
```
criterion = nn.BCEWithLogitsLoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
```
In each epoch of the training loop, we first train the discriminator, by taking real and fake images.
Then we train the generator multiple times in each epoch.


