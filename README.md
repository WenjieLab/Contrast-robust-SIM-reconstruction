# CR-SIM
The code is developed for CR-SIM reconstruction and is related to the paper "Deep learning enables contrast-robust super-resolution reconstruction in structured illumination microscopy." (will be updated soon)

# User Guide
### Environment Set-up
- Install Anaconda ([Learn more](https://docs.anaconda.com/anaconda/install/))

- Download the repo from Github:
  `https://github.com/WenjieLab/Contrast-robust-SIM-reconstruction.git`

- Create a conda environment for pssr:
  `conda create --name crsim python=3.7`

- Activate the conda environment:
  `conda activate crsim`

- Install crsim dependencies:
  `pip install -r requirements.txt`

### Scenario 1: Inference using our pretrained models
- Download pre-trained models of CR-SIM and place them in ```./weight/```
- Open your terminal and run `python predict.py` in your terminal. Note that before running the python file, you should check if the data paths and other arguments in predict.py are set correctly
- The output SR images will be saved in ```./testdata/raw/output_raw-model/```

### Scenario 2: Train your own data
- Prepare the training dataset: You can train a new CR-SIM model using your own datasets. Note that you'd better divide the dataset of each specimen into training part and validation/testing part before training, so that you can test your model with the preserved validation/testing data
- Run `python train.py` in your terminal to train a new CR-SIM model. Similar to testing, before running the python file, you should check if the data paths and the arguments are set correctly
- Model weights will be saved in `../trained_models_Data` by default

# Model Structure
![image](https://github.com/WenjieLab/Contrast-robust-SIM-reconstruction/assets/52398597/cb8c0d18-b10d-40b4-8dc7-dc9ad3510fa2) <br>
Our CR-SIM network is based on Residual U-Net, which consists of an Encoder-Decoder structure with skip connections. The combination of U-Net and residual allows the CR-SIM network to be a reliable and efficient solution for super-resolution SIM reconstruction. <br>
### (1)	Encoder:<br>
The encoder contains four downsampling blocks, each consisting of a 1x1 convolutional layer, a 3x3 convolutional layer with ReLU activation, a dropout layer, and another 3x3 convolutional layer with ReLU activation. Skip connections are established by adding the output of the 1x1 convolution to the output of the second 3x3 convolution in each block. MaxPooling2D with a 2x2 pool size is applied to reduce spatial dimensions, and the number of features is doubled after each block.<br>
### (2)	Decoder:<br>
The decoder consists of attention-based upsampling blocks that concatenate features from the encoder's corresponding skip connection. Each upsampling block involves transposed convolution with a scale factor of 2, concatenation of skip connection features, and two sets of 3x3 convolutions with ReLU activations. The number of features is halved after each block.<br>
