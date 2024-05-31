module add_biases(out, in, biases, clk, rst);
parameter IN_CHANNELS = 1,
					IN_SIZE = 3,
					FILTERS = 1,
					KER_SIZE = 3,
					STRIDE = 1,
					PADDING = 0,
					NON_ZERO_WEIGHTS=6,
					BIT_SIZE = 16,
					OUTPUT_SIZE = (((IN_SIZE - KER_SIZE + 2*PADDING)/STRIDE)+1);

output reg [((OUTPUT_SIZE**2)*FILTERS*BIT_SIZE)-1:0] out;
input [((OUTPUT_SIZE**2)*FILTERS*BIT_SIZE)-1:0] in;
input [FILTERS*BIT_SIZE-1:0] biases;

input clk, rst;

// Internal parameters
wire [((OUTPUT_SIZE**2)*FILTERS*BIT_SIZE)-1:0] out_wire;
wire [((OUTPUT_SIZE**2)*BIT_SIZE)-1:0] channels [FILTERS-1:0];

// Instantiate 16-bit CSeLA's to add biases
generate
genvar j,k;
for (k=0; k<FILTERS; k=k+1) begin
    for (j=0; j<(OUTPUT_SIZE**2); j=j+1) begin : CSELA
        carrySelectAdder16bit csela_adder(.sum(out_wire[k*((OUTPUT_SIZE**2)*BIT_SIZE)+j*BIT_SIZE +: BIT_SIZE]), .in1(in[k*((OUTPUT_SIZE**2)*BIT_SIZE)+j*BIT_SIZE +: BIT_SIZE]), .in2(biases[k*BIT_SIZE +: BIT_SIZE]));
    end
end
endgenerate

integer l;
always @ (posedge clk) begin
if(!rst) begin
out <= 0;
end
else begin

// Apply ReLU activation function after adding biases
for (l=0; l<(OUTPUT_SIZE**2)*FILTERS*BIT_SIZE; l=l+BIT_SIZE) begin
    if (out_wire [l + (BIT_SIZE-1)] == 0)
		    out [l +: BIT_SIZE] <= out_wire [l +: BIT_SIZE];
    else 
				out [l +: BIT_SIZE] <= 0;
end

end
end

endmodule
