# CR-SIM
The code is developed for CR-SIM reconstruction and is related to the paper "Deep learning enables contrast-robust super-resolution reconstruction in structured illumination microscopy." (will be updated soon)

# Environment
GPU: NVIDIA GeForce RTX 2060 <br>
Keras 2.3.1 <br>
CUDA 11.1 <br>
Python 3.7 <br>

# Model Structure
![image](https://github.com/WenjieLab/Contrast-robust-SIM-reconstruction/assets/52398597/cb8c0d18-b10d-40b4-8dc7-dc9ad3510fa2) <br>
Our CR-SIM network is based on Residual U-Net, which consists of an Encoder-Decoder structure with skip connections. The combination of U-Net and residual allows the CR-SIM network to be a reliable and efficient solution for super-resolution SIM reconstruction. <br>
(1)	Encoder:<br>
The encoder contains four downsampling blocks, each consisting of a 1x1 convolutional layer, a 3x3 convolutional layer with ReLU activation, a dropout layer, and another 3x3 convolutional layer with ReLU activation. Skip connections are established by adding the output of the 1x1 convolution to the output of the second 3x3 convolution in each block. MaxPooling2D with a 2x2 pool size is applied to reduce spatial dimensions, and the number of features is doubled after each block.<br>
(2)	Decoder:<br>
The decoder consists of attention-based upsampling blocks that concatenate features from the encoder's corresponding skip connection. Each upsampling block involves transposed convolution with a scale factor of 2, concatenation of skip connection features, and two sets of 3x3 convolutions with ReLU activations. The number of features is halved after each block.<br>
