### Conv 2D
Makes a kernel that slides along two dimensions. So suppose we have a 24x24 RGB image, we have an
input which is a 24x24x3 tensor. If we write 
```nn.Conv2D(in_channels=3, out_channels=12, kernel_size=5, stride=1)```
This produces a 12 (out_channels) kernels each 5x5x3 (kernel_size x kernel_size x in_channels) in
dimension. Each of these will convolve with the tensor with stride 1 to give output of dimension 20x20x1. 
These 12 outputs are then stacked to give a tensor of size 20x20x12
