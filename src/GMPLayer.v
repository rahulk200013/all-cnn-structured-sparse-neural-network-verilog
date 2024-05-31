module gmp_layer(out, in, clk, rst);
parameter BIT_SIZE = 16,
					ACTIVATIONS_GMP = 36,
					NUM_CLASSES = 10;

output reg [NUM_CLASSES*BIT_SIZE-1:0] out;
input [ACTIVATIONS_GMP*NUM_CLASSES*BIT_SIZE-1:0] in;

input clk, rst;

wire [NUM_CLASSES*BIT_SIZE-1:0] final_out;

// Instantiate global max pool module for each input channel
generate
genvar i;
		for (i=0; i<NUM_CLASSES; i=i+1) begin : gmp
				global_maxpool #(.BIT_SIZE(BIT_SIZE), .ACTIVATIONS_GMP(ACTIVATIONS_GMP)) gmp(in[i*ACTIVATIONS_GMP*BIT_SIZE +: ACTIVATIONS_GMP*BIT_SIZE], final_out[i*BIT_SIZE +: BIT_SIZE], clk, rst);
		end
endgenerate

always @ (posedge clk) begin
		if (!rst) out <= 0;
		else out <= final_out;
end

endmodule
