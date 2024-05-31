module cnn_layer(out_act, in_act, weights, biases, index, r_pointer, clk, rst);
parameter IN_CHANNELS = 3,
					IN_SIZE = 32,
					FILTERS = 1,
					KER_SIZE = 3,
					STRIDE = 1,
					PADDING = 0,
					NON_ZERO_WEIGHTS=6,
					BIT_SIZE = 16,
					FRACTIONAL_BITS = 8,
					INDEX_BIT_SIZE = 3,
					R_POINTER_BIT_SIZE = 3,
					SPECIAL_CASE = 0,
					OUTPUT_SIZE = (((IN_SIZE - KER_SIZE + 2*PADDING)/STRIDE)+1);

output reg [((OUTPUT_SIZE**2)*FILTERS*BIT_SIZE)-1:0] out_act;
input [(IN_SIZE**2)*IN_CHANNELS*BIT_SIZE-1:0] in_act;
input [NON_ZERO_WEIGHTS*FILTERS*BIT_SIZE-1:0] weights;
input [FILTERS*BIT_SIZE-1:0] biases;
input [NON_ZERO_WEIGHTS*INDEX_BIT_SIZE-1:0] index;
input [IN_CHANNELS*KER_SIZE*R_POINTER_BIT_SIZE-1:0] r_pointer;

input clk, rst;

wire [((OUTPUT_SIZE**2)*BIT_SIZE)-1:0] vgm_out_act;
wire [FILTERS*BIT_SIZE-1:0] vgm_out_weights;
wire [((OUTPUT_SIZE**2)*FILTERS*BIT_SIZE)-1:0] conv_out, layer_out;


// ##################################################################################################################################
// ###################################                      Convolutional Layer                     #################################
// ##################################################################################################################################
vector_generator #(.IN_CHANNELS(IN_CHANNELS),
             .IN_SIZE(IN_SIZE),
             .FILTERS(FILTERS),
             .KER_SIZE(KER_SIZE),
             .STRIDE(STRIDE),
             .PADDING(PADDING),
             .NON_ZERO_WEIGHTS(NON_ZERO_WEIGHTS),
             .BIT_SIZE(BIT_SIZE),
	     .INDEX_BIT_SIZE(INDEX_BIT_SIZE),
	     .R_POINTER_BIT_SIZE(R_POINTER_BIT_SIZE)) vgm (vgm_out_act, vgm_out_weights, in_act, weights, index, r_pointer, clk, rst);

conv_layer #(.IN_CHANNELS(IN_CHANNELS),
             .IN_SIZE(IN_SIZE),
             .FILTERS(FILTERS),
             .KER_SIZE(KER_SIZE),
             .STRIDE(STRIDE),
             .PADDING(PADDING),
             .NON_ZERO_WEIGHTS(NON_ZERO_WEIGHTS),
             .BIT_SIZE(BIT_SIZE),
						 .FRACTIONAL_BITS(FRACTIONAL_BITS),
	     .SPECIAL_CASE(SPECIAL_CASE)) conv (conv_out, vgm_out_act, vgm_out_weights, clk, rst);

add_biases #(.IN_CHANNELS(IN_CHANNELS),
             .IN_SIZE(IN_SIZE),
             .FILTERS(FILTERS),
             .KER_SIZE(KER_SIZE),
             .STRIDE(STRIDE),
             .PADDING(PADDING),
             .NON_ZERO_WEIGHTS(NON_ZERO_WEIGHTS),
	     .BIT_SIZE(BIT_SIZE)) addBiases (layer_out, conv_out, biases, clk, rst);

// ##################################################################################################################################
// ##################################################################################################################################
// ##################################################################################################################################

always @ (posedge clk) begin
if(!rst)
out_act <= 0;

else
out_act <= layer_out;

end
endmodule
