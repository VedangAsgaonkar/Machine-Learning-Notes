To decide between which of Conv1D, Conv2D and Conv3D to use, see in which dimensions are spacial features important. In an rgb image, the tensor is 3D, but spacial features are present only in 2 dimensions, so we use Conv2D. If we are processing a signal, we will use Conv1D as only spacial features are important along the time dimension. To process a rgb video which is 4 dimensional tensor, we use Conv3D as spacial features are important along the width, height and time

### Conv2d
Makes a kernel that slides along two dimensions. So suppose we have a 24x24 RGB image, we have an
input which is a 3x24x24 tensor. If we write 
```nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1)```
This produces a 12 (out_channels) kernels each 5x5x3 (in_channels x kernel_size x kernel_size) in
dimension. Each of these will convolve with the tensor with stride 1 to give output of dimension 1x20x20. 
These 12 outputs are then stacked to give a tensor of size 12x20x20. We can additionaly specify the padding in the method as ```padding=1```
<br>
Note that the common convention is to represent images as height x width x color, but PyTorch uses color x height x width. Most datasets are in the former format, and the generator ```torch.utils.data.DataLoader``` converts it to the latter format

### BatchNorm2d
The batch normalization layer is used to make sure that activations do not have a very large shift, allowing for faster convergence. [Deep learning basics — batch normalization](https://medium.com/analytics-vidhya/deep-learning-basics-batch-normalization-ae105f9f537e#:~:text=BatchNorm2d%20is%20the%20number%20of,to%20the%20batch%20norm%20layer.). 
```
nn.BatchNorm2d(32)
```
performs batch normalisation along the width and height dimensions for the data taking data in batches of 32.

### MaxPool2d
```
nn.MaxPool2d(2)
```
performs max pooling in squares of size 2 x 2
