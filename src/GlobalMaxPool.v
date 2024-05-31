module global_maxpool(in, out, clk, rst);
parameter BIT_SIZE=8,     // No of bits of each pixel in the feature map
          ACTIVATIONS_GMP=36;  // No of pixels in each channel of 6x6 size

// Input and Output ports
output reg signed [BIT_SIZE-1:0] out;
input wire signed [BIT_SIZE*ACTIVATIONS_GMP-1:0] in;
input clk, rst;

integer j;
always @ (posedge clk) begin
		// Reset
		if(!rst) begin
				out <= 0;
		end

		else begin
				// Set output to zero with blocking assignment
				out = 0;

				// Iterate through each input and compare it with last max value
				for (j = 0; j < ACTIVATIONS_GMP; j = j + 1) begin
						if (in[j*BIT_SIZE +: BIT_SIZE] > out)
							 out <= in[j*BIT_SIZE +: BIT_SIZE];
				end
		end
end
endmodule
