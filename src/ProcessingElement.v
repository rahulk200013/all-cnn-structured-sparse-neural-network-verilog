module processing_element(activation, weight, out, clk, rst);
parameter NON_ZERO_WEIGHTS = 27,
					SPECIAL_CASE = 0,
					FRACTIONAL_BITS = 8,
					BIT_SIZE = 16;

output reg signed [BIT_SIZE-1:0] out;
input [BIT_SIZE-1:0] activation, weight;
input clk, rst;

wire signed [26:0] out_27bit;
reg signed [26:0] shifted_out;
reg [BIT_SIZE-1:0] activation_reg, weight_reg;

// Instantiate 16-bit MAC to perform convolution
mac_16bit #(.NON_ZERO_WEIGHTS(NON_ZERO_WEIGHTS), .FRACTIONAL_BITS(FRACTIONAL_BITS)) mac0 (out_27bit,activation_reg,weight_reg,clk,rst);

always @ (posedge clk) begin
		if(!rst) begin
				activation_reg <= 0;
				weight_reg <= 0;
				out <= 0;
				shifted_out <= 0;
		end

		else begin
		activation_reg <= activation;
		weight_reg <= weight;
		// if (SPECIAL_CASE) begin
				// shifted_out = out_27bit >>> 1;
				// $display("%b", out_27bit); 
				// out <= (shifted_out[BIT_SIZE-1:0]);
		// end
		// else
		
		// Assign 16 Least significiant bits to output
		// Currently the BIT_SIZE can't be other than 16 as MAC is made for 16 bits only
		out <= out_27bit[BIT_SIZE-1:0];
		end
end
endmodule


