module max_layer(out, in, clk, rst);
parameter BIT_SIZE=8,     // No of bits in each input
          NUM_CLASSES=10;  // Total no of inputs

// Input and Output ports
output reg [$clog2(NUM_CLASSES)-1:0] out;
input [BIT_SIZE*NUM_CLASSES-1:0] in;
input clk, rst;

// Internal variables
reg [$clog2(NUM_CLASSES)-1:0] count;
reg [$clog2(NUM_CLASSES)-1:0] temp_out;
reg [BIT_SIZE-1:0] max_val;


always @ (posedge clk) begin
		// Reset
		if(!rst) begin
				count <= 0;
				max_val <= 0;
				out <= 0;
				temp_out <= 0;
		end
		else begin
				if (count >= 0) begin
						// Iterate through each value to find the max value
						if (in[count*BIT_SIZE +: BIT_SIZE] > max_val) begin
								max_val <= in[count*BIT_SIZE +: BIT_SIZE];
								temp_out <= count;
						end
				end
				// After iterating through every input value, update the output 
				// with the class having the max probability
				if (count == NUM_CLASSES) begin
						out <= NUM_CLASSES - 1 - temp_out;  // Classes are in reverse order and hence it is subtracted from 9 to get the correct class
						
						// Reset counter and max value for next operation
						max_val <= 0;
						count <= 0;
				end
				else count <= count + 1;
		end
end
endmodule
