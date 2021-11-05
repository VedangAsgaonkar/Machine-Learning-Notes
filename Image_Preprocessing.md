### Reading images from a directory
```
train_dataset = datasets.ImageFolder(
  'data/train',
  tranform = train_transform
)
```
### Making Image Transforms
The simplest transform, which is *required* is ```torchvision.transforms.ToTensor()```.
Other transforms the can be specified seperately are done as
```
train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.CenterCrop(size=224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```
