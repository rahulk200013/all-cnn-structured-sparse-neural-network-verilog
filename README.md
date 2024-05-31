# ASIC Implementation of All CNN Structured Sparse Neural Network in Verilog

This project demonstrates the ASIC implementation of an All CNN Structured Sparse Neural Network in Verilog, based on the following papers:
- ["An Efficient Hardware Accelerator for Structured Sparse Convolutional Neural Networks on FPGAs"](https://arxiv.org/abs/2001.01955)
- ["Striving for Simplicity: The All Convolutional Net"](https://arxiv.org/abs/1412.6806)

## Features
- Fully parameterized architecture with customizable layers:
  - Input Channels
  - Input Size
  - Number of Kernels
  - Kernel Size
  - Stride
  - Padding
- 16-bit fixed-point arithmetic (8 integer bits, 8 fractional bits).
- Convolutional layers only; dense layers must be converted to 1x1 convolutional layers.
- Global max pooling after convolutional layers.
- Default configuration set for the MNIST dataset.

## Prerequisites
- Python
- PyTorch
- Verilog

## Training and Saving the Model
Train your model in PyTorch with the given constraints and save it as follows:
```python
torch.save({'model': model.state_dict(),
            ...
           },
           'cnn_model.pt')
```

## Running this project
- Clone the repository:
  ``` bash
  git clone https://github.com/rahulk200013/all-cnn-structured-sparse-neural-network-verilog.git
  cd all-cnn-structured-sparse-neural-network-verilog
  ```
- Place the trained model `cnn_model.pt` in the `model` folder.
- Modify the CNN architecture in scripts/cnn_model_arch.py to match your model's parameters.
- If using a dataset other than MNIST, update the dataset details in scripts/generate_testbench.py at line 223.
- Generate updated verilog files:
  ``` bash
  python scripts/generate_verilog_and_LUTs.py --in_size=<image_size> --num_classes=<num_classes>
  python scripts/generate_testbench.py --num_classes=<num_classes>
  ```
  For example, for the MNIST dataset with an image size of 28x28 and 10 classes:
  ``` bash
  python scripts/generate_verilog_and_LUTs.py --in_size=28 --num_classes=10
  python scripts/generate_testbench.py --num_classes=10
  ```
- Replace the following generated files with those in the src folder:
  - `cnn_model.v`
  - `testbench.v`
  - `lut_weights.v`
  - `lut_biases.v`
  - `lut_index.v`
  - `lut_rpointer.v`
 
## References
- [An Efficient Hardware Accelerator for Structured Sparse Convolutional Neural Networks on FPGAs](https://arxiv.org/abs/2001.01955)
- [Striving for Simplicity: The All Convolutional Net](https://arxiv.org/abs/1412.6806)




