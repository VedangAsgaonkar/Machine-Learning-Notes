In transfer learning, we make use of weights from pre-trained models that are known to work well.
We add a few layers above these frozen weights and train only that part of the model. This is particularly useful
in computer vision where we keep the "feature extractor" i.e. the CNN and add our own fc layers on top. This is
because for models trained on huge datasets like ImageNet, the "feature extractor" is adept at identifying important
features.
<br>
Some typical models whose weights are used in transfer learning are VGG, ResNet(which also has parallel kernels), Inception.
<br>
We can take two approaches to use the inherited weights :
* Since the weights do not change during training, we can pre-compute the output of all inputs through these instead of computing in every epoch. This has the dissadvantage that we cannot augment the data
* Use data augmentation, and use the weights in every epoch
