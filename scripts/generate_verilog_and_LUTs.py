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
from math import floor
import argparse

from collections import OrderedDict
from cnn_model_arch import cnn_model


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

# This code is contributed by Ryuga 

# Function to find two's complement
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


def binaryToDecimal(binary):
 
    decimal, i = 0, 0
    while(binary != 0):
        dec = binary % 10
        decimal = decimal + dec * pow(2, i)
        binary = binary//10
        i += 1
    return decimal

def ones_complement(binary):
    list1 = list(binary)
    for i in range(len(binary)):
        if (list1[i] == "0"):
            list1[i] = "1"
        else:
            list1[i] = "0"

    return ''.join(list1)

def bitExtend(binary, integer_bits):

    remaining_bits = integer_bits - len(binary)

    extra_str = ""

    if remaining_bits>0:
        for i in range(remaining_bits):
            extra_str = extra_str + "0"

    return extra_str+binary

def check_sparsity(tensor):
    zeros = 0
    total = len(tensor)

    for i in range(total):
        if(tensor[i] == 0):
            zeros += 1

    return (zeros/total)

def model_depth(network):
    layers = 0
    for v in network.modules():
        if isinstance(v, torch.nn.modules.conv.Conv2d):
            layers += 1

    return layers


def check_acc(model, loader):
    device = torch.device("cpu")
      
    correct = 0
    total = 0
    
    model.eval()
    
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
            
    return accuracy


def main():

    parser = argparse.ArgumentParser(description='All CNN Structured Sparse Neural Network Verilog Generator script')
    parser.add_argument('--in_size', action="store", default=28)
    parser.add_argument('--num_classes', action="store", default=10)

    args = parser.parse_args()

    in_size = int(args.in_size)
    num_classes = int(args.num_classes)

    total_bits = 16
    fractional_bits = 8
    integer_bits = total_bits - fractional_bits

    index_bit_size = 3
    r_pointer_bit_size = 3

    binary_sel = []

    device = torch.device("cpu")
    model = cnn_model()
    model.to(device)

    conv_layers = model_depth(model)

    sel_bits = math.ceil(math.log(conv_layers+1, 2))

    for i in range(conv_layers):
        binary_sel.append(bitExtend(decimalToBinary(i+1, 0).replace(".",""), sel_bits))

    load_model = 'model/cnn_model.pt' 
    print('==> Loading from {}'.format(load_model))
    print('\n==> Generating all LUTs...')
    state_dict = torch.load(load_model, map_location='cpu')

    # train_transform = transforms.Compose([transforms.ToTensor()])

    # trainset = torchvision.datasets.MNIST(root='./data', train=False,
    #                                         download=True, transform=train_transform)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
    #                                           shuffle=False, num_workers=4)

    batch_size = 1

    # testset = datasets.MNIST('./data', train=False, download=True, transform=transforms.ToTensor())  # MNIST
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

    new_state_dict = state_dict['model'] 
    model.load_state_dict(state_dict['model'])  # MNIST 

    binary_weights = []
    weights_error = []
    non_zero_weigths_list = []

    small_count = 0
    torch.set_printoptions(sci_mode=False, precision=20)

    layer_count = 1

    for k, v in new_state_dict.items():
        print(k)
        weights_str = ""
        if('weight' in k):
            flattened_weight = v.flatten()
            non_zero_weight = flattened_weight[flattened_weight.nonzero()]
            non_zero_weigths_list.append(int(len(non_zero_weight)/getattr(model,'conv'+str(layer_count)).out_channels))
            layer_count += 1

            for i in range(len(flattened_weight)):
                binary, decimal, error = convert_to_fixed_point(flattened_weight[i], integer_bits, fractional_bits)
                weights_str += binary
                weights_error.append(abs(error))

            binary_weights.append(weights_str)


    f = open("lut_weights.v", "w")
    for i in range(conv_layers):
        # f.write(str(sel_bits) + "'b" + binary_sel[i] + ": sbyte = " + str(len(binary_weights[i])) + "'b" + binary_weights[i] + ";\n" )
        f.write("module lut_weights_" + str(i+1)+ "(sbyte,addr);\n")
        f.write("input [" + str(sel_bits-1) + ":0] addr;\n")
        f.write("output reg [" + str() + str(len(binary_weights[i])-1) + ":0] sbyte;\n\n")

        f.write("always @ (addr) begin\n\n")

        f.write("(* synthesis, full_case, parallel_case *) case (addr)\n\n")

        f.write(str(sel_bits) + "'b" + binary_sel[i] + ": sbyte = " + str(len(binary_weights[i])) + "'b" + binary_weights[i] + ";\n" )

        f.write("endcase\n")
        f.write("end\n")
        f.write("endmodule\n\n\n\n")
    f.close()


    binary_biases = []
    biases_error = []

    for k, v in new_state_dict.items():
        biases_str = ""
        if('bias' in k):
            flattened_bias = v.flatten()

            for i in range(len(flattened_bias)):
                binary, decimal, error = convert_to_fixed_point(flattened_bias[i], integer_bits, fractional_bits)
                biases_str += binary
                biases_error.append(abs(error))

            binary_biases.append(biases_str)


    f = open("lut_biases.v", "w")
    for i in range(conv_layers):
        f.write("module lut_biases_" + str(i+1)+ "(sbyte,addr);\n")
        f.write("input [" + str(sel_bits-1) + ":0] addr;\n")
        f.write("output reg [" + str() + str(len(binary_biases[i])-1) + ":0] sbyte;\n\n")

        f.write("always @ (addr) begin\n\n")

        f.write("(* synthesis, full_case, parallel_case *) case (addr)\n\n")

        f.write(str(sel_bits) + "'b" + binary_sel[i] + ": sbyte = " + str(len(binary_biases[i])) + "'b" + binary_biases[i] + ";\n" )

        f.write("endcase\n")
        f.write("end\n")
        f.write("endmodule\n\n\n\n")

    f.close()

    # KER_SIZE = 3
    # KER_SIZE = getattr(model,'conv1').kernel_size[0]

    index_binary = []
    r_pointer_binary = []

    layer_count = 1

    for k, v in new_state_dict.items():
        index_str = ""
        r_pointer_str = ""

        index = []
        r_pointer = []

        if ('weight' in k):
            kernel = v[0]

            flattened = kernel.flatten()
            non_zero_weigths_index = flattened.nonzero()

            count = 0
            KER_SIZE = getattr(model,'conv' + str(layer_count)).kernel_size[0]
            layer_count += 1

            for i in range(int(len(flattened)/KER_SIZE)):
                nonzero_weights_in_row = 0
                first_non_zero_weight_in_row = 1
                for j in range(KER_SIZE):
                    if (flattened[KER_SIZE*i + j] != 0):
                        nonzero_weights_in_row += 1
                        if (j==0):
                            index.append(int(non_zero_weigths_index[count] - KER_SIZE*i))
                            first_non_zero_weight_in_row = 0
                            count += 1
                        else:
                            if (first_non_zero_weight_in_row):
                                index.append(int(non_zero_weigths_index[count] - KER_SIZE*i))
                                first_non_zero_weight_in_row = 0
                            else:
                                index.append(int(non_zero_weigths_index[count] - non_zero_weigths_index[count-1] - 1))
                            count += 1
                r_pointer.append(nonzero_weights_in_row)

            for i in range(len(index)):
                index_str += convert_to_fixed_point(index[i], index_bit_size, 0)[0]

            index_binary.append(index_str)

            for i in range(len(r_pointer)):
                r_pointer_str += convert_to_fixed_point(r_pointer[i], r_pointer_bit_size, 0)[0]

            r_pointer_binary.append(r_pointer_str)

    f = open("lut_index.v", "w")
    for i in range(conv_layers):
        f.write("module lut_index_" + str(i+1)+ "(sbyte,addr);\n")
        f.write("input [" + str(sel_bits-1) + ":0] addr;\n")
        f.write("output reg [" + str() + str(len(index_binary[i])-1) + ":0] sbyte;\n\n")

        f.write("always @ (addr) begin\n\n")

        f.write("(* synthesis, full_case, parallel_case *) case (addr)\n\n")

        f.write(str(sel_bits) + "'b" + binary_sel[i] + ": sbyte = " + str(len(index_binary[i])) + "'b" + index_binary[i] + ";\n" )

        f.write("endcase\n")
        f.write("end\n")
        f.write("endmodule\n\n\n\n")

    f.close()

    f = open("lut_r_pointer.v", "w")
    for i in range(conv_layers):
        f.write("module lut_r_pointer_" + str(i+1)+ "(sbyte,addr);\n")
        f.write("input [" + str(sel_bits-1) + ":0] addr;\n")
        f.write("output reg [" + str() + str(len(r_pointer_binary[i])-1) + ":0] sbyte;\n\n")

        f.write("always @ (addr) begin\n\n")

        f.write("(* synthesis, full_case, parallel_case *) case (addr)\n\n")

        f.write(str(sel_bits) + "'b" + binary_sel[i] + ": sbyte = " + str(len(r_pointer_binary[i])) + "'b" + r_pointer_binary[i] + ";\n" )

        f.write("endcase\n")
        f.write("end\n")
        f.write("endmodule\n\n\n\n")

    f.close()

    

    print('==> Generating all LUTs DONE')
    print('\nMaximum errors after converting to binary:')
    print("Max error in weights: ", max(weights_error).item())
    print("Max error in biases: ", max(biases_error).item())

    
    # summary(model, input_size=(1, 28, 28))

    

    print("\n==> Generating verilog file:")

    layers = model_depth(model)

    layer_param = {}

    for i in range(layers):
        layer_param["layer_{0}".format(i+1)] = []

    for i in range(layers):
        in_channels = getattr(model,'conv'+str(i+1)).in_channels
        filters = getattr(model,'conv'+str(i+1)).out_channels
        ker_size = getattr(model,'conv'+str(i+1)).kernel_size[0]
        stride = getattr(model,'conv'+str(i+1)).stride[0]
        padding = getattr(model,'conv'+str(i+1)).padding[0]
        non_zero_weights = non_zero_weigths_list[i]
        output_size = floor(((in_size - ker_size + 2*padding)/stride) + 1)

        layer_param["layer_" + str(i+1)].extend((in_channels, in_size, filters, ker_size, stride, padding, non_zero_weights))

        in_size = output_size

    print("Convolutional Layer parameters:")
    for k,v in layer_param.items():
        print(k,v)

    activations_gmp = in_size*in_size

    f = open("cnn_model.v", "w")
    f.write("module model_cnn(in, out, clk, rst);\n")
    f.write("parameter BIT_SIZE = 16,\n")
    f.write("          FRACTIONAL_BITS = 8,\n")
    f.write("          INDEX_BIT_SIZE = 3,\n")
    f.write("          R_POINTER_BIT_SIZE = 3,\n")
    f.write("          NUM_CLASSES = " + str(num_classes) + ",\n")
    f.write("          ACTIVATIONS_GMP = " + str(activations_gmp) + ",\n\n\n")

    for i in range(layers):
        f.write("          // Layer " + str(i+1) + " Parameters\n")
        f.write("          IN_CHANNELS_" + str(i+1) + " = " + str(layer_param['layer_' + str(i+1)][0]) + ",\n")
        f.write("          IN_SIZE_" + str(i+1) + " = " + str(layer_param['layer_' + str(i+1)][1]) + ",\n")
        f.write("          FILTERS_" + str(i+1) + " = " + str(layer_param['layer_' + str(i+1)][2]) + ",\n")
        f.write("          KER_SIZE_" + str(i+1) + " = " + str(layer_param['layer_' + str(i+1)][3]) + ",\n")
        f.write("          STRIDE_" + str(i+1) + " = " + str(layer_param['layer_' + str(i+1)][4]) + ",\n")
        f.write("          PADDING_" + str(i+1) + " = " + str(layer_param['layer_' + str(i+1)][5]) + ",\n")
        f.write("          NON_ZERO_WEIGHTS_" + str(i+1) + " = " + str(layer_param['layer_' + str(i+1)][6]) + ",\n")
        if (i == layers - 1):
            f.write("          OUTPUT_SIZE_" + str(i+1) + " = " + "(((IN_SIZE_" + str(i+1) + " - KER_SIZE_" + str(i+1) + " + 2*PADDING_" + str(i+1) + ")/STRIDE_" + str(i+1) + ")+1);\n\n\n")
        else:
            f.write("          OUTPUT_SIZE_" + str(i+1) + " = " + "(((IN_SIZE_" + str(i+1) + " - KER_SIZE_" + str(i+1) + " + 2*PADDING_" + str(i+1) + ")/STRIDE_" + str(i+1) + ")+1),\n\n\n")

    f.write("output reg [3:0] out;\n")
    f.write("input [(IN_SIZE_1**2)*IN_CHANNELS_1*BIT_SIZE-1:0] in;\n\n")
    f.write("input clk, rst;\n\n")
    f.write("// Wires for weights coming from LUTs\n")
    for i in range(layers):
        f.write("wire [NON_ZERO_WEIGHTS_" + str(i+1) + "*FILTERS_" + str(i+1) + "*BIT_SIZE-1:0] weights_" + str(i+1) + ";\n")

    f.write("\n// Wires for biases coming from LUTs\n")
    for i in range(layers):
        f.write("wire [FILTERS_" + str(i+1) + "*BIT_SIZE-1:0] biases_" + str(i+1) + ";\n")

    f.write("\n// Wires for index values coming from LUTs\n")
    for i in range(layers):
        f.write("wire [NON_ZERO_WEIGHTS_" + str(i+1) + "*INDEX_BIT_SIZE-1:0] index_" + str(i+1) + ";\n")

    f.write("\n// Wires for r_pointer values coming from LUTs\n")
    for i in range(layers):
        f.write("wire [KER_SIZE_" + str(i+1) + "*IN_CHANNELS_" + str(i+1) + "*R_POINTER_BIT_SIZE-1:0] r_pointer_" + str(i+1) + ";\n")

    f.write("\n// Select parameter to choose the correct layer parameters from LUTs\n")
    f.write("reg [3:0] select [8:0];\n\n")

    f.write("// Get all the weights from LUTs\n")
    for i in range(layers):
        f.write("lut_weights_" + str(i+1) + " w" + str(i+1) + "(weights_" + str(i+1) + ", select[" + str(i) + "]);\n")

    f.write("\n// Get all the biases from LUTs\n")
    for i in range(layers):
        f.write("lut_biases_" + str(i+1) + " b" + str(i+1) + "(biases_" + str(i+1) + ", select[" + str(i) + "]);\n")

    f.write("\n// Get all the index values from LUTs\n")
    for i in range(layers):
        f.write("lut_index_" + str(i+1) + " i" + str(i+1) + "(index_" + str(i+1) + ", select[" + str(i) + "]);\n")

    f.write("\n// Get all the r-pointer values from LUTs\n")
    for i in range(layers):
        f.write("lut_r_pointer_" + str(i+1) + " r" + str(i+1) + "(r_pointer_" + str(i+1) + ", select[" + str(i) + "]);\n")

    f.write("\n// Wires for output of each layer\n")
    for i in range(layers):
        f.write("wire [((OUTPUT_SIZE_" + str(i+1) + "**2)*FILTERS_" + str(i+1) + "*BIT_SIZE)-1:0] layer_out_" + str(i+1) + ";\n")

    f.write("\n// Wire for output from Global Max Pool layer\n")
    f.write("wire [NUM_CLASSES*BIT_SIZE-1:0] gmp_out;\n")

    f.write("\n// Wire for final output\n")
    f.write("wire [3:0] final_out;\n\n")

    for i in range(layers):
        f.write("\n// ##################################################################################################################################\n")
        f.write("// ###################################                      Layer " + str(i+1) + "                                 #################################\n")
        f.write("// ##################################################################################################################################\n")
        f.write("cnn_layer #(.IN_CHANNELS(IN_CHANNELS_" + str(i+1) + "),\n")
        f.write("            .IN_SIZE(IN_SIZE_" + str(i+1) + "),\n")
        f.write("            .FILTERS(FILTERS_" + str(i+1) + "),\n")
        f.write("            .KER_SIZE(KER_SIZE_" + str(i+1) + "),\n")
        f.write("            .STRIDE(STRIDE_" + str(i+1) + "),\n")
        f.write("            .PADDING(PADDING_" + str(i+1) + "),\n")
        f.write("            .NON_ZERO_WEIGHTS(NON_ZERO_WEIGHTS_" + str(i+1) + "),\n")
        f.write("            .BIT_SIZE(BIT_SIZE),\n")
        f.write("            .FRACTIONAL_BITS(FRACTIONAL_BITS),\n")
        f.write("            .INDEX_BIT_SIZE(INDEX_BIT_SIZE),\n")
        if (i == 0):
            f.write("            .R_POINTER_BIT_SIZE(R_POINTER_BIT_SIZE)) layer_" + str(i+1) + " (layer_out_" + str(i+1) + ", in, weights_" + str(i+1) + ", biases_" + str(i+1) + ", index_" + str(i+1) + ", r_pointer_" + str(i+1) + ", clk, rst);\n")
        else:
            f.write("            .R_POINTER_BIT_SIZE(R_POINTER_BIT_SIZE)) layer_" + str(i+1) + " (layer_out_" + str(i+1) + ", layer_out_" + str(i) + ", weights_" + str(i+1) + ", biases_" + str(i+1) + ", index_" + str(i+1) + ", r_pointer_" + str(i+1) + ", clk, rst);\n")

    f.write("\n// ##################################################################################################################################\n")
    f.write("// ###################################                      Global Max Pool                         #################################\n")
    f.write("// ##################################################################################################################################\n")
    f.write("gmp_layer #(.BIT_SIZE(BIT_SIZE),\n")
    f.write("            .NUM_CLASSES(NUM_CLASSES),\n")
    f.write("            .ACTIVATIONS_GMP(ACTIVATIONS_GMP)) gmp (gmp_out, layer_out_" + str(layers) + ", clk, rst);\n\n")

    f.write("// ##################################################################################################################################\n")
    f.write("// ###################################                      CLASSIFICATION                          #################################\n")
    f.write("// ##################################################################################################################################\n")
    f.write("max_layer #(.BIT_SIZE(BIT_SIZE),\n")
    f.write("            .NUM_CLASSES(NUM_CLASSES)) max (final_out, gmp_out, clk, rst);\n\n")

    f.write("always @ (posedge clk) begin\n")
    f.write("    if (!rst) begin\n")
    f.write("        out <= 0;\n")
    for i in range(layers):
        f.write("        select[" + str(i) + "] <= " + str(i+1) + ";\n")
    f.write("    end\n")
    f.write("    else out <= final_out;\n")
    f.write("end\n\n")
    f.write("endmodule\n")


    f.close()

    print("==> Generating verilog file DONE")
    print("Verilog file has been saved as cnn_mnist.v")

    print("Finished Successfully")








if __name__ == "__main__":
    main()
