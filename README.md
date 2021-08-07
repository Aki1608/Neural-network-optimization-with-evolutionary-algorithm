# Neural-network-optimization-using-evolutionary-algorithm

Optimization of a neural network which is based on U-Net. This network is optimized for the task of semantic segmentation. It follows the structure of U-Net (encoder-decoder type network structure) but the layers will be selected randomly and the network will be optimized accordingly. 

## Installation 
Use the package manager [pip](https://pip.pypa.io/en/stable/) to install required libraries. All the required libraries are listed in requirements.txt file. You can install each libraries separately. For example

```bash
pip install opencv-python
pip install numpy
```

or you can install all at once with 

```bash
pip install -r requirements.txt
```

## Usage
Run
```bash
python Main.py
```

## Working

Running Main.py will first of all create a list of network parameters and then according to this list networks will be generated as per `train and accuracy` function in `train.py`. After generating and training network, an excel sheet with the network architecture information will be generated, which includes almost everything from layer, their parameters like number of kernel, feature maps, memory used and number of operation used at each stage. With this information a cost function will be caculated and top and some random netowrks will be selected according to constraints and networks for next generation will be generated. This procedure will repeat untill we reach to the last generation. 

You will notice that networks will start optimizing rapidly according to our constraints in starting generations. Optimization will reach to saturation at later generations.
