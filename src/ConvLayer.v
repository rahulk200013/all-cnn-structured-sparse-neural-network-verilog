module conv_layer(out, activations, weights, clk, rst);
parameter IN_CHANNELS = 3,
					IN_SIZE = 32,
					FILTERS = 48,
					KER_SIZE = 3,
					STRIDE = 1,
					PADDING = 1,
					NON_ZERO_WEIGHTS=27,
					BIT_SIZE = 16,
					FRACTIONAL_BITS = 8,
					SPECIAL_CASE = 0,
					LAYER = 0;

output reg [((((((IN_SIZE - KER_SIZE + 2*PADDING)/STRIDE)+1)**2)*FILTERS)*BIT_SIZE)-1 : 0] out;
input [(FILTERS*BIT_SIZE)-1 : 0] weights;
input [(((((IN_SIZE - KER_SIZE + 2*PADDING)/STRIDE)+1)**2)*BIT_SIZE)-1:0] activations;

input clk, rst;

reg [$clog2(NON_ZERO_WEIGHTS):0] weight_index;  // Index to select weight. It goes from 0 to NON_ZERO_WEIGHTS-1
reg [4:0] setup_cycles;                         // Clock cycle to wait for modules like VGM to start giving outputs.

// Internal variables
wire [BIT_SIZE-1:0] wire_out [((((IN_SIZE - KER_SIZE + 2*PADDING)/STRIDE)+1)**2)*FILTERS-1:0]; 
wire [((((IN_SIZE - KER_SIZE + 2*PADDING)/STRIDE)+1)**2)*FILTERS*BIT_SIZE-1:0] final_out;
reg [((((IN_SIZE - KER_SIZE + 2*PADDING)/STRIDE)+1)**2)*FILTERS*BIT_SIZE-1:0] reg_final_out;
reg [BIT_SIZE-1:0] reg_act [((((IN_SIZE - KER_SIZE + 2*PADDING)/STRIDE)+1)**2)*FILTERS-1:0];
reg [BIT_SIZE-1:0] reg_weights [((((IN_SIZE - KER_SIZE + 2*PADDING)/STRIDE)+1)**2)*FILTERS-1:0];

// Instantiate all the required processing elements as per the layer parameters
generate
genvar i,l;
for(i=0; i<((((IN_SIZE - KER_SIZE + 2*PADDING)/STRIDE)+1)**2)*FILTERS; i=i+1) begin : PE
processing_element #(.NON_ZERO_WEIGHTS(NON_ZERO_WEIGHTS), .SPECIAL_CASE(SPECIAL_CASE), .FRACTIONAL_BITS(FRACTIONAL_BITS), .BIT_SIZE(BIT_SIZE)) pe (.activation(reg_act[i]), .weight(reg_weights[i]), .out(wire_out[i]), .clk(clk), .rst(rst));
end

// Convert 2D output to 1D output so that we can pass it on to add bias module
for (l=0; l<((((IN_SIZE - KER_SIZE + 2*PADDING)/STRIDE)+1)**2)*FILTERS; l=l+1) begin
    assign final_out[(l*BIT_SIZE) +: BIT_SIZE] = wire_out[l];
end
endgenerate

integer j,k,m;
always @ (posedge clk) begin
		// Reset module. Reset is active low.
		if (!rst) begin
				out <= 0;
				weight_index <= 0;
				reg_final_out <= 0;
				setup_cycles <= 0;
				for (k=0; k<((((IN_SIZE - KER_SIZE + 2*PADDING)/STRIDE)+1)**2)*FILTERS; k=k+1) begin
						reg_act[k] <= 0;
						reg_weights[k] <= 0;
				end
		end
		else begin
				// Update output registor with new value
				reg_final_out <= final_out;
				
				// Update activations and weights register with new activations and weights recieved from VGM
				for (k=0; k<FILTERS; k=k+1) begin
						for(j=0; j<((((IN_SIZE - KER_SIZE + 2*PADDING)/STRIDE)+1)**2); j=j+1) begin
								reg_act[k*((((IN_SIZE - KER_SIZE + 2*PADDING)/STRIDE)+1)**2) + j] <= activations[BIT_SIZE*j +: BIT_SIZE];
								reg_weights[k*((((IN_SIZE - KER_SIZE + 2*PADDING)/STRIDE)+1)**2) + j] <= weights[BIT_SIZE*k +: BIT_SIZE];
						end
				end
				// Wait until VGM setup is completed and then start the operation.
				// It takes 7 clock cycles to recieve first input from VGM
				// We need to wait as every layer needs to work in sync to work properly.
				if (setup_cycles == 7) begin
						// Update output after every NON_ZERO_WEIGHTS cycles
						// This prevents all the intermediate garbage values to reflect in the output
						// Only the final output is updated in the output registor
						if (weight_index == NON_ZERO_WEIGHTS-1) begin
								weight_index <= 0;
								out <= reg_final_out;
						end
						else weight_index <= weight_index + 1;
				end
				else setup_cycles <= setup_cycles + 1;
		end
end
endmodule
