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

### Pre-processing
When we feed in images to a transfer learning model, we have to be careful about the data being of the same format as the data in which the model was 
trained on. For example, vgg16 requires the following transforms to be applied, besides other transforms that we may choose to apply
```
vgg_transforms = torchvision.transforms.Compose([
  transforms.Resize(size=256),
  transforms.CenterCrop(size=224),
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```
### Loading pre-trained models
```
model = models.vgg16(pretrained=True)

# Freeze VGG weights
for param in model.parameters():
  param.requires_grad = False
```
### Editing pre-trained models
vgg16 has an fc layer called classifier. We can redefine the fc layer as
```
model.classifier = nn.Linear(25088, 2)
```
Similarly, we can add our own layers on top
