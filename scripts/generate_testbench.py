from __future__ import print_function
import math
from binary_fractions import Binary
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
import numpy as np
from numpy import linalg as LA
import sys
from math import floor, log2

from collections import OrderedDict
from cnn_model_arch import cnn_model

def debug_model(model, loader):
    device = torch.device("cpu")

    model.eval()

    count = 0
    input_imgs, predicted_classes = [], []

    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 0)
            input_imgs.append(labels.item())
            predicted_classes.append(predicted.item())
            count += 1
            if (count == 5):
                print("Test images: ", input_imgs)
                print("Prediction:  ", predicted_classes)
                break

def convert_to_fixed_point(num, integer_bits, fractional_bits):
    
    if (num >= 0):
        binary = decimalToBinary(num, fractional_bits).replace(".", "")
        final_binary = signExtend('0' + binary, integer_bits, fractional_bits, 0)

    else:
        abs_num = abs(num)
        binary = decimalToBinary(abs_num, fractional_bits).replace(".","")
        twos_complement = findTwoscomplement(binary)
        if (len(twos_complement) > len(binary)):
            final_binary = signExtend('0' + twos_complement[1:len(twos_complement)-1], integer_bits, fractional_bits, 1)
        else:
            final_binary = signExtend('1' + twos_complement, integer_bits, fractional_bits, 1)

    decimal = parse_bin(num, final_binary, integer_bits, fractional_bits)
    error = (abs(num) - abs(decimal))

    return final_binary, decimal, error

def signExtend(binary, integer_bits, fractional_bits, sign):

    remaining_bits = (integer_bits + fractional_bits) - len(binary)

    extra_bits_str = ""
    final_binary = ""

    if (remaining_bits < 0):
        print("Binary exceeds " + str(integer_bits + fractional_bits) + " bits size")

        if( sign == 0 ):
            print("Setting +ve max value")
            return "0111111111111111"
        else:
            print("Setting -ve max value")
            print(len(binary))
            return "1000000000000000"

    elif (remaining_bits >= 0):
      for i in range(remaining_bits):
          extra_bits_str += binary[0]
      final_binary = extra_bits_str + binary
      final_binary = final_binary.replace(".", "")

      return final_binary

def findTwoscomplement(str):
    n = len(str)
 
    # Traverse the string to get first 
    # '1' from the last of string
    i = n - 1
    while(i >= 0):
        if (str[i] == '1'):
            break
 
        i -= 1
 
    # If there exists no '1' concatenate 1 
    # at the starting of string
    if (i == -1):
        return '1'+str
 
    # Continue traversal after the 
    # position of first '1'
    k = i - 1
    while(k >= 0):
         
        # Just flip the values
        if (str[k] == '1'):
            str = list(str)
            str[k] = '0'
            str = ''.join(str)
        else:
            str = list(str)
            str[k] = '1'
            str = ''.join(str)
 
        k -= 1
 
    # return the modified string
    return str

def decimalToBinary(num, k_prec) : 

    binary = "" 

    # Fetch the integral part of 
    # decimal number 
    Integral = int(num) 

    # Fetch the fractional part 
    # decimal number 
    fractional = num - Integral 

    # Conversion of integral part to 
    # binary equivalent 
    while (Integral) : 
        
        rem = Integral % 2

        # Append 0 in binary 
        binary += str(rem); 

        Integral //= 2
    
    # Reverse string to get original 
    # binary equivalent 
    binary = binary[ : : -1] 

    # Append point before conversion 
    # of fractional part 
    binary += '.'

    # Conversion of fractional part 
    # to binary equivalent 
    while (k_prec) : 
        
        # Find next bit in fraction 
        fractional *= 2
        fract_bit = int(fractional) 

        if (fract_bit == 1) : 
            
            fractional -= fract_bit 
            binary += '1'
            
        else : 
            binary += '0'

        k_prec -= 1

    return binary 

def parse_bin(num, binary, integer_bits, fractional_bits):
    if (num < 0):
        binary = findTwoscomplement(binary)
    integer = binary[len(binary) - (integer_bits + fractional_bits) :integer_bits]
    fraction = binary[integer_bits:integer_bits + fractional_bits]
    # print("Integer part: ", integer)
    # print("Fractional part: ", fraction)
    if (fractional_bits > 0):
        out = int(integer, 2) + int(fraction, 2) / 2.**len(fraction)
    else:
        out = 0
    if (num < 0):
        return -out
    else:
        return out

def main():

    parser = argparse.ArgumentParser(description='All CNN Structured Sparse Neural Network Verilog Generator script')
    parser.add_argument('--num_classes', action="store", default=10)

    args = parser.parse_args()

    num_classes = int(args.num_classes)

    total_bits = 16
    fractional_bits = 8
    integer_bits = total_bits - fractional_bits

    device = torch.device("cpu")
    model = cnn_model()   # CIFAR10 latest
    model.to(device)

    load_model = 'model/cnn_model.pt'

    print('==> Loading from {}'.format(load_model))

    state_dict = torch.load(load_model, map_location='cpu')
    model.load_state_dict(state_dict['model'])  # MNIST 
    batch_size = 1

    ######################################################################################################################
    ##########                                 CHANGE YOUR DATASET HERE                                         ##########
    ######################################################################################################################

    testset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())  # MNIST
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=4)

    ######################################################################################################################
    ######################################################################################################################
    ######################################################################################################################

    count = 0

    print("\n==> Generating 5 random input images binary: ")

    # f = open("input_img_binary.txt", "w")
    error_list = []
    labels_list = []
    prediction_list = []
    binary_inputs = []

    model.eval()

    for data in testloader:
        img_str = ""
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        flattened_img = images[0].flatten()
        for i in range(len(flattened_img)):
            binary, decimal, error = convert_to_fixed_point(flattened_img[i], integer_bits, fractional_bits)
            img_str += binary
            error_list.append(abs(error))

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 0)
        labels_list.append(labels.item())
        prediction_list.append(predicted.item())

        # f.write("Label: " + str(labels[0]))
        # f.write("\n\n" + img_str)
        # f.write("\n\n\n")

        binary_inputs.append(img_str)

        # print("Label: ", labels[0].item())
        count += 1
        if (count == 5):
            break

    # f.close()

    # debug_model(model, testloader)
    print("Labels: ", labels_list)
    print("Prediction: ", prediction_list)
    print("Max error in input: ", max(error_list).item())
    print("DONE")
    
    print("\n==> Generating Testbench")

    f = open("testbench.v", "w")
    f.write("`timescale 1ps/1fs\n\n")
    f.write("module test_cnn();\n\n")
    f.write("reg [" + str(len(img_str)-1) + ":0] in;\n")
    f.write("wire [" + str(floor(log2(num_classes))) + ":0] out;\n\n")
    f.write("reg clk, rst;\n\n")
    f.write("// Instantiate CNN module\n")
    f.write("model_cnn cnn (in, out, clk, rst);\n\n")
    f.write("always #5 clk = ~clk;\n\n")
    f.write("initial begin\n")
    f.write("    // Display output in terminal if there is any change in its value\n")
    f.write('    $monitor("Predicted Class: %d", out);\n\n')
    f.write("    clk = 1;\n")
    f.write("    rst = 0;  // Send active low reset signal to every module\n\n")
    f.write("    #50\n")
    f.write("    rst = 1;\n\n")
    f.write("    // Uncomment any one of the input to test\n\n")
    for i in range(5):
        f.write("    // " + str(labels_list[i]) + "\n")
        f.write("    // in = " + str(len(img_str)) + "'b" + binary_inputs[i] + "\n\n")

    f.write("end")
    f.write("endmodule")

    f.close()

    print("DONE")




if __name__ == "__main__":
    main()
