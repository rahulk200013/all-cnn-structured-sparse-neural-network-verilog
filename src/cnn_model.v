module model_cnn(in, out, clk, rst);
parameter BIT_SIZE = 16,
          FRACTIONAL_BITS = 8,
          INDEX_BIT_SIZE = 3,
          R_POINTER_BIT_SIZE = 3,
          NUM_CLASSES = 10,
          ACTIVATIONS_GMP = 25,


          // Layer 1 Parameters
          IN_CHANNELS_1 = 1,
          IN_SIZE_1 = 28,
          FILTERS_1 = 12,
          KER_SIZE_1 = 3,
          STRIDE_1 = 1,
          PADDING_1 = 1,
          NON_ZERO_WEIGHTS_1 = 9,
          OUTPUT_SIZE_1 = (((IN_SIZE_1 - KER_SIZE_1 + 2*PADDING_1)/STRIDE_1)+1),


          // Layer 2 Parameters
          IN_CHANNELS_2 = 12,
          IN_SIZE_2 = 28,
          FILTERS_2 = 12,
          KER_SIZE_2 = 3,
          STRIDE_2 = 1,
          PADDING_2 = 1,
          NON_ZERO_WEIGHTS_2 = 108,
          OUTPUT_SIZE_2 = (((IN_SIZE_2 - KER_SIZE_2 + 2*PADDING_2)/STRIDE_2)+1),


          // Layer 3 Parameters
          IN_CHANNELS_3 = 12,
          IN_SIZE_3 = 28,
          FILTERS_3 = 12,
          KER_SIZE_3 = 3,
          STRIDE_3 = 2,
          PADDING_3 = 1,
          NON_ZERO_WEIGHTS_3 = 108,
          OUTPUT_SIZE_3 = (((IN_SIZE_3 - KER_SIZE_3 + 2*PADDING_3)/STRIDE_3)+1),


          // Layer 4 Parameters
          IN_CHANNELS_4 = 12,
          IN_SIZE_4 = 14,
          FILTERS_4 = 24,
          KER_SIZE_4 = 3,
          STRIDE_4 = 1,
          PADDING_4 = 1,
          NON_ZERO_WEIGHTS_4 = 108,
          OUTPUT_SIZE_4 = (((IN_SIZE_4 - KER_SIZE_4 + 2*PADDING_4)/STRIDE_4)+1),


          // Layer 5 Parameters
          IN_CHANNELS_5 = 24,
          IN_SIZE_5 = 14,
          FILTERS_5 = 24,
          KER_SIZE_5 = 3,
          STRIDE_5 = 1,
          PADDING_5 = 1,
          NON_ZERO_WEIGHTS_5 = 216,
          OUTPUT_SIZE_5 = (((IN_SIZE_5 - KER_SIZE_5 + 2*PADDING_5)/STRIDE_5)+1),


          // Layer 6 Parameters
          IN_CHANNELS_6 = 24,
          IN_SIZE_6 = 14,
          FILTERS_6 = 24,
          KER_SIZE_6 = 3,
          STRIDE_6 = 2,
          PADDING_6 = 1,
          NON_ZERO_WEIGHTS_6 = 216,
          OUTPUT_SIZE_6 = (((IN_SIZE_6 - KER_SIZE_6 + 2*PADDING_6)/STRIDE_6)+1),


          // Layer 7 Parameters
          IN_CHANNELS_7 = 24,
          IN_SIZE_7 = 7,
          FILTERS_7 = 36,
          KER_SIZE_7 = 3,
          STRIDE_7 = 1,
          PADDING_7 = 0,
          NON_ZERO_WEIGHTS_7 = 216,
          OUTPUT_SIZE_7 = (((IN_SIZE_7 - KER_SIZE_7 + 2*PADDING_7)/STRIDE_7)+1),


          // Layer 8 Parameters
          IN_CHANNELS_8 = 36,
          IN_SIZE_8 = 5,
          FILTERS_8 = 36,
          KER_SIZE_8 = 1,
          STRIDE_8 = 1,
          PADDING_8 = 0,
          NON_ZERO_WEIGHTS_8 = 36,
          OUTPUT_SIZE_8 = (((IN_SIZE_8 - KER_SIZE_8 + 2*PADDING_8)/STRIDE_8)+1),


          // Layer 9 Parameters
          IN_CHANNELS_9 = 36,
          IN_SIZE_9 = 5,
          FILTERS_9 = 10,
          KER_SIZE_9 = 1,
          STRIDE_9 = 1,
          PADDING_9 = 0,
          NON_ZERO_WEIGHTS_9 = 36,
          OUTPUT_SIZE_9 = (((IN_SIZE_9 - KER_SIZE_9 + 2*PADDING_9)/STRIDE_9)+1);


output reg [$clog2(NUM_CLASSES)-1:0] out;
input [(IN_SIZE_1**2)*IN_CHANNELS_1*BIT_SIZE-1:0] in;

input clk, rst;

// Wires for weights coming from LUTs
wire [NON_ZERO_WEIGHTS_1*FILTERS_1*BIT_SIZE-1:0] weights_1;
wire [NON_ZERO_WEIGHTS_2*FILTERS_2*BIT_SIZE-1:0] weights_2;
wire [NON_ZERO_WEIGHTS_3*FILTERS_3*BIT_SIZE-1:0] weights_3;
wire [NON_ZERO_WEIGHTS_4*FILTERS_4*BIT_SIZE-1:0] weights_4;
wire [NON_ZERO_WEIGHTS_5*FILTERS_5*BIT_SIZE-1:0] weights_5;
wire [NON_ZERO_WEIGHTS_6*FILTERS_6*BIT_SIZE-1:0] weights_6;
wire [NON_ZERO_WEIGHTS_7*FILTERS_7*BIT_SIZE-1:0] weights_7;
wire [NON_ZERO_WEIGHTS_8*FILTERS_8*BIT_SIZE-1:0] weights_8;
wire [NON_ZERO_WEIGHTS_9*FILTERS_9*BIT_SIZE-1:0] weights_9;

// Wires for biases coming from LUTs
wire [FILTERS_1*BIT_SIZE-1:0] biases_1;
wire [FILTERS_2*BIT_SIZE-1:0] biases_2;
wire [FILTERS_3*BIT_SIZE-1:0] biases_3;
wire [FILTERS_4*BIT_SIZE-1:0] biases_4;
wire [FILTERS_5*BIT_SIZE-1:0] biases_5;
wire [FILTERS_6*BIT_SIZE-1:0] biases_6;
wire [FILTERS_7*BIT_SIZE-1:0] biases_7;
wire [FILTERS_8*BIT_SIZE-1:0] biases_8;
wire [FILTERS_9*BIT_SIZE-1:0] biases_9;

// Wires for index values coming from LUTs
wire [NON_ZERO_WEIGHTS_1*INDEX_BIT_SIZE-1:0] index_1;
wire [NON_ZERO_WEIGHTS_2*INDEX_BIT_SIZE-1:0] index_2;
wire [NON_ZERO_WEIGHTS_3*INDEX_BIT_SIZE-1:0] index_3;
wire [NON_ZERO_WEIGHTS_4*INDEX_BIT_SIZE-1:0] index_4;
wire [NON_ZERO_WEIGHTS_5*INDEX_BIT_SIZE-1:0] index_5;
wire [NON_ZERO_WEIGHTS_6*INDEX_BIT_SIZE-1:0] index_6;
wire [NON_ZERO_WEIGHTS_7*INDEX_BIT_SIZE-1:0] index_7;
wire [NON_ZERO_WEIGHTS_8*INDEX_BIT_SIZE-1:0] index_8;
wire [NON_ZERO_WEIGHTS_9*INDEX_BIT_SIZE-1:0] index_9;

// Wires for r_pointer values coming from LUTs
wire [KER_SIZE_1*IN_CHANNELS_1*R_POINTER_BIT_SIZE-1:0] r_pointer_1;
wire [KER_SIZE_2*IN_CHANNELS_2*R_POINTER_BIT_SIZE-1:0] r_pointer_2;
wire [KER_SIZE_3*IN_CHANNELS_3*R_POINTER_BIT_SIZE-1:0] r_pointer_3;
wire [KER_SIZE_4*IN_CHANNELS_4*R_POINTER_BIT_SIZE-1:0] r_pointer_4;
wire [KER_SIZE_5*IN_CHANNELS_5*R_POINTER_BIT_SIZE-1:0] r_pointer_5;
wire [KER_SIZE_6*IN_CHANNELS_6*R_POINTER_BIT_SIZE-1:0] r_pointer_6;
wire [KER_SIZE_7*IN_CHANNELS_7*R_POINTER_BIT_SIZE-1:0] r_pointer_7;
wire [KER_SIZE_8*IN_CHANNELS_8*R_POINTER_BIT_SIZE-1:0] r_pointer_8;
wire [KER_SIZE_9*IN_CHANNELS_9*R_POINTER_BIT_SIZE-1:0] r_pointer_9;

// Select parameter to choose the correct layer parameters from LUTs
reg [3:0] select [8:0];

// Get all the weights from LUTs
lut_weights_1 w1(weights_1, select[0]);
lut_weights_2 w2(weights_2, select[1]);
lut_weights_3 w3(weights_3, select[2]);
lut_weights_4 w4(weights_4, select[3]);
lut_weights_5 w5(weights_5, select[4]);
lut_weights_6 w6(weights_6, select[5]);
lut_weights_7 w7(weights_7, select[6]);
lut_weights_8 w8(weights_8, select[7]);
lut_weights_9 w9(weights_9, select[8]);

// Get all the biases from LUTs
lut_biases_1 b1(biases_1, select[0]);
lut_biases_2 b2(biases_2, select[1]);
lut_biases_3 b3(biases_3, select[2]);
lut_biases_4 b4(biases_4, select[3]);
lut_biases_5 b5(biases_5, select[4]);
lut_biases_6 b6(biases_6, select[5]);
lut_biases_7 b7(biases_7, select[6]);
lut_biases_8 b8(biases_8, select[7]);
lut_biases_9 b9(biases_9, select[8]);

// Get all the index values from LUTs
lut_index_1 i1(index_1, select[0]);
lut_index_2 i2(index_2, select[1]);
lut_index_3 i3(index_3, select[2]);
lut_index_4 i4(index_4, select[3]);
lut_index_5 i5(index_5, select[4]);
lut_index_6 i6(index_6, select[5]);
lut_index_7 i7(index_7, select[6]);
lut_index_8 i8(index_8, select[7]);
lut_index_9 i9(index_9, select[8]);

// Get all the r-pointer values from LUTs
lut_r_pointer_1 r1(r_pointer_1, select[0]);
lut_r_pointer_2 r2(r_pointer_2, select[1]);
lut_r_pointer_3 r3(r_pointer_3, select[2]);
lut_r_pointer_4 r4(r_pointer_4, select[3]);
lut_r_pointer_5 r5(r_pointer_5, select[4]);
lut_r_pointer_6 r6(r_pointer_6, select[5]);
lut_r_pointer_7 r7(r_pointer_7, select[6]);
lut_r_pointer_8 r8(r_pointer_8, select[7]);
lut_r_pointer_9 r9(r_pointer_9, select[8]);

// Wires for output of each layer
wire [((OUTPUT_SIZE_1**2)*FILTERS_1*BIT_SIZE)-1:0] layer_out_1;
wire [((OUTPUT_SIZE_2**2)*FILTERS_2*BIT_SIZE)-1:0] layer_out_2;
wire [((OUTPUT_SIZE_3**2)*FILTERS_3*BIT_SIZE)-1:0] layer_out_3;
wire [((OUTPUT_SIZE_4**2)*FILTERS_4*BIT_SIZE)-1:0] layer_out_4;
wire [((OUTPUT_SIZE_5**2)*FILTERS_5*BIT_SIZE)-1:0] layer_out_5;
wire [((OUTPUT_SIZE_6**2)*FILTERS_6*BIT_SIZE)-1:0] layer_out_6;
wire [((OUTPUT_SIZE_7**2)*FILTERS_7*BIT_SIZE)-1:0] layer_out_7;
wire [((OUTPUT_SIZE_8**2)*FILTERS_8*BIT_SIZE)-1:0] layer_out_8;
wire [((OUTPUT_SIZE_9**2)*FILTERS_9*BIT_SIZE)-1:0] layer_out_9;

// Wire for output from Global Max Pool layer
wire [NUM_CLASSES*BIT_SIZE-1:0] gmp_out;

// Wire for final output
wire [3:0] final_out;


// ##################################################################################################################################
// ###################################                      Layer 1                                 #################################
// ##################################################################################################################################
cnn_layer #(.IN_CHANNELS(IN_CHANNELS_1),
            .IN_SIZE(IN_SIZE_1),
            .FILTERS(FILTERS_1),
            .KER_SIZE(KER_SIZE_1),
            .STRIDE(STRIDE_1),
            .PADDING(PADDING_1),
            .NON_ZERO_WEIGHTS(NON_ZERO_WEIGHTS_1),
            .BIT_SIZE(BIT_SIZE),
            .FRACTIONAL_BITS(FRACTIONAL_BITS),
            .INDEX_BIT_SIZE(INDEX_BIT_SIZE),
            .R_POINTER_BIT_SIZE(R_POINTER_BIT_SIZE)) layer_1 (layer_out_1, in, weights_1, biases_1, index_1, r_pointer_1, clk, rst);

// ##################################################################################################################################
// ###################################                      Layer 2                                 #################################
// ##################################################################################################################################
cnn_layer #(.IN_CHANNELS(IN_CHANNELS_2),
            .IN_SIZE(IN_SIZE_2),
            .FILTERS(FILTERS_2),
            .KER_SIZE(KER_SIZE_2),
            .STRIDE(STRIDE_2),
            .PADDING(PADDING_2),
            .NON_ZERO_WEIGHTS(NON_ZERO_WEIGHTS_2),
            .BIT_SIZE(BIT_SIZE),
            .FRACTIONAL_BITS(FRACTIONAL_BITS),
            .INDEX_BIT_SIZE(INDEX_BIT_SIZE),
            .R_POINTER_BIT_SIZE(R_POINTER_BIT_SIZE)) layer_2 (layer_out_2, layer_out_1, weights_2, biases_2, index_2, r_pointer_2, clk, rst);

// ##################################################################################################################################
// ###################################                      Layer 3                                 #################################
// ##################################################################################################################################
cnn_layer #(.IN_CHANNELS(IN_CHANNELS_3),
            .IN_SIZE(IN_SIZE_3),
            .FILTERS(FILTERS_3),
            .KER_SIZE(KER_SIZE_3),
            .STRIDE(STRIDE_3),
            .PADDING(PADDING_3),
            .NON_ZERO_WEIGHTS(NON_ZERO_WEIGHTS_3),
            .BIT_SIZE(BIT_SIZE),
            .FRACTIONAL_BITS(FRACTIONAL_BITS),
            .INDEX_BIT_SIZE(INDEX_BIT_SIZE),
            .R_POINTER_BIT_SIZE(R_POINTER_BIT_SIZE)) layer_3 (layer_out_3, layer_out_2, weights_3, biases_3, index_3, r_pointer_3, clk, rst);

// ##################################################################################################################################
// ###################################                      Layer 4                                 #################################
// ##################################################################################################################################
cnn_layer #(.IN_CHANNELS(IN_CHANNELS_4),
            .IN_SIZE(IN_SIZE_4),
            .FILTERS(FILTERS_4),
            .KER_SIZE(KER_SIZE_4),
            .STRIDE(STRIDE_4),
            .PADDING(PADDING_4),
            .NON_ZERO_WEIGHTS(NON_ZERO_WEIGHTS_4),
            .BIT_SIZE(BIT_SIZE),
            .FRACTIONAL_BITS(FRACTIONAL_BITS),
            .INDEX_BIT_SIZE(INDEX_BIT_SIZE),
            .R_POINTER_BIT_SIZE(R_POINTER_BIT_SIZE)) layer_4 (layer_out_4, layer_out_3, weights_4, biases_4, index_4, r_pointer_4, clk, rst);

// ##################################################################################################################################
// ###################################                      Layer 5                                 #################################
// ##################################################################################################################################
cnn_layer #(.IN_CHANNELS(IN_CHANNELS_5),
            .IN_SIZE(IN_SIZE_5),
            .FILTERS(FILTERS_5),
            .KER_SIZE(KER_SIZE_5),
            .STRIDE(STRIDE_5),
            .PADDING(PADDING_5),
            .NON_ZERO_WEIGHTS(NON_ZERO_WEIGHTS_5),
            .BIT_SIZE(BIT_SIZE),
            .FRACTIONAL_BITS(FRACTIONAL_BITS),
            .INDEX_BIT_SIZE(INDEX_BIT_SIZE),
            .R_POINTER_BIT_SIZE(R_POINTER_BIT_SIZE)) layer_5 (layer_out_5, layer_out_4, weights_5, biases_5, index_5, r_pointer_5, clk, rst);

// ##################################################################################################################################
// ###################################                      Layer 6                                 #################################
// ##################################################################################################################################
cnn_layer #(.IN_CHANNELS(IN_CHANNELS_6),
            .IN_SIZE(IN_SIZE_6),
            .FILTERS(FILTERS_6),
            .KER_SIZE(KER_SIZE_6),
            .STRIDE(STRIDE_6),
            .PADDING(PADDING_6),
            .NON_ZERO_WEIGHTS(NON_ZERO_WEIGHTS_6),
            .BIT_SIZE(BIT_SIZE),
            .FRACTIONAL_BITS(FRACTIONAL_BITS),
            .INDEX_BIT_SIZE(INDEX_BIT_SIZE),
            .R_POINTER_BIT_SIZE(R_POINTER_BIT_SIZE)) layer_6 (layer_out_6, layer_out_5, weights_6, biases_6, index_6, r_pointer_6, clk, rst);

// ##################################################################################################################################
// ###################################                      Layer 7                                 #################################
// ##################################################################################################################################
cnn_layer #(.IN_CHANNELS(IN_CHANNELS_7),
            .IN_SIZE(IN_SIZE_7),
            .FILTERS(FILTERS_7),
            .KER_SIZE(KER_SIZE_7),
            .STRIDE(STRIDE_7),
            .PADDING(PADDING_7),
            .NON_ZERO_WEIGHTS(NON_ZERO_WEIGHTS_7),
            .BIT_SIZE(BIT_SIZE),
            .FRACTIONAL_BITS(FRACTIONAL_BITS),
            .INDEX_BIT_SIZE(INDEX_BIT_SIZE),
            .R_POINTER_BIT_SIZE(R_POINTER_BIT_SIZE)) layer_7 (layer_out_7, layer_out_6, weights_7, biases_7, index_7, r_pointer_7, clk, rst);

// ##################################################################################################################################
// ###################################                      Layer 8                                 #################################
// ##################################################################################################################################
cnn_layer #(.IN_CHANNELS(IN_CHANNELS_8),
            .IN_SIZE(IN_SIZE_8),
            .FILTERS(FILTERS_8),
            .KER_SIZE(KER_SIZE_8),
            .STRIDE(STRIDE_8),
            .PADDING(PADDING_8),
            .NON_ZERO_WEIGHTS(NON_ZERO_WEIGHTS_8),
            .BIT_SIZE(BIT_SIZE),
            .FRACTIONAL_BITS(FRACTIONAL_BITS),
            .INDEX_BIT_SIZE(INDEX_BIT_SIZE),
            .R_POINTER_BIT_SIZE(R_POINTER_BIT_SIZE)) layer_8 (layer_out_8, layer_out_7, weights_8, biases_8, index_8, r_pointer_8, clk, rst);

// ##################################################################################################################################
// ###################################                      Layer 9                                 #################################
// ##################################################################################################################################
cnn_layer #(.IN_CHANNELS(IN_CHANNELS_9),
            .IN_SIZE(IN_SIZE_9),
            .FILTERS(FILTERS_9),
            .KER_SIZE(KER_SIZE_9),
            .STRIDE(STRIDE_9),
            .PADDING(PADDING_9),
            .NON_ZERO_WEIGHTS(NON_ZERO_WEIGHTS_9),
            .BIT_SIZE(BIT_SIZE),
            .FRACTIONAL_BITS(FRACTIONAL_BITS),
            .INDEX_BIT_SIZE(INDEX_BIT_SIZE),
            .R_POINTER_BIT_SIZE(R_POINTER_BIT_SIZE)) layer_9 (layer_out_9, layer_out_8, weights_9, biases_9, index_9, r_pointer_9, clk, rst);

// ##################################################################################################################################
// ###################################                      Global Max Pool                         #################################
// ##################################################################################################################################
gmp_layer #(.BIT_SIZE(BIT_SIZE),
            .NUM_CLASSES(NUM_CLASSES),
            .ACTIVATIONS_GMP(ACTIVATIONS_GMP)) gmp (gmp_out, layer_out_9, clk, rst);

// ##################################################################################################################################
// ###################################                      CLASSIFICATION                          #################################
// ##################################################################################################################################
max_layer #(.BIT_SIZE(BIT_SIZE),
            .NUM_CLASSES(NUM_CLASSES)) max (final_out, gmp_out, clk, rst);

always @ (posedge clk) begin
    if (!rst) begin
        out <= 0;
        select[0] <= 1;
        select[1] <= 2;
        select[2] <= 3;
        select[3] <= 4;
        select[4] <= 5;
        select[5] <= 6;
        select[6] <= 7;
        select[7] <= 8;
        select[8] <= 9;
    end
    else out <= final_out;
end

endmodule
