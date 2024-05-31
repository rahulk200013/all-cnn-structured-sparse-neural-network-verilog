module vector_generator(out_activations, out_weights, in_activations, in_weights, index, r_pointer, clk, rst);
parameter IN_CHANNELS = 1,
					IN_SIZE = 3,
					FILTERS = 2,
					KER_SIZE = 3,
					STRIDE = 1,
					PADDING = 0,
					NON_ZERO_WEIGHTS=6,
					BIT_SIZE = 16,
					INDEX_BIT_SIZE = 3,
					R_POINTER_BIT_SIZE = 3;

output reg [(((((IN_SIZE-KER_SIZE+(2*PADDING))/STRIDE)+1)**2)*BIT_SIZE)-1:0] out_activations;
output reg [FILTERS*BIT_SIZE-1:0] out_weights;
input [((IN_SIZE**2)*IN_CHANNELS*BIT_SIZE)-1:0] in_activations;
input [FILTERS*NON_ZERO_WEIGHTS*BIT_SIZE-1:0] in_weights;
input [NON_ZERO_WEIGHTS*INDEX_BIT_SIZE-1:0] index;
input [(KER_SIZE*IN_CHANNELS*R_POINTER_BIT_SIZE)-1:0] r_pointer;

input clk, rst;

// Internal variables
reg [((IN_SIZE**2)*IN_CHANNELS*BIT_SIZE)-1:0] reg_in_activations;
reg [$clog2(NON_ZERO_WEIGHTS):0] weight_index, conv_act_index, weight_sel_index, operations_count;
reg [INDEX_BIT_SIZE-1:0] start_pos;
reg calc_index_pos, setup_completed, vgm_setup_completed;
reg [BIT_SIZE-1:0] act_index [NON_ZERO_WEIGHTS-1:0];

wire [INDEX_BIT_SIZE-1:0] index_2d [(NON_ZERO_WEIGHTS)-1:0];
wire [R_POINTER_BIT_SIZE-1:0] r_pointer_2d [(KER_SIZE*IN_CHANNELS)-1:0];
wire [(((IN_SIZE+(2*PADDING))**2)*IN_CHANNELS*BIT_SIZE)-1:0] in_act_padded;
wire [(KER_SIZE**2)*IN_CHANNELS*BIT_SIZE-1 : 0] conv_wind [((((IN_SIZE - KER_SIZE + 2*PADDING)/STRIDE)+1)**2)-1 : 0]; 

//Perform Padding on input activation
generate
genvar x, y, z;
for (z=0; z<IN_CHANNELS; z=z+1) begin
		// Add zero row above and below the input tensor
    for (x=0; x<PADDING; x=x+1) begin
        assign in_act_padded[z*((IN_SIZE+(2*PADDING))**2)*BIT_SIZE  + x*(IN_SIZE+(2*PADDING))*BIT_SIZE +: (IN_SIZE+(2*PADDING))*BIT_SIZE] = 0;
        assign in_act_padded[((((IN_SIZE+(2*PADDING))**2)*IN_CHANNELS*BIT_SIZE)-1) - ((IN_CHANNELS-1-z)*(((IN_SIZE+(2*PADDING))**2)*BIT_SIZE) + x*(IN_SIZE+(2*PADDING))*BIT_SIZE) -: (IN_SIZE+(2*PADDING))*BIT_SIZE] = 0;
    end
		// Add zero before and after the each row of input tensor
    for (x=0; x<IN_SIZE; x=x+1) begin
        if(PADDING) begin  
            assign in_act_padded[z*((IN_SIZE+(2*PADDING))**2)*BIT_SIZE + (PADDING+x)*(IN_SIZE+(2*PADDING))*BIT_SIZE +: PADDING*BIT_SIZE] = 0;
            assign in_act_padded[z*((IN_SIZE+(2*PADDING))**2)*BIT_SIZE + (PADDING+x+1)*(IN_SIZE+(2*PADDING))*BIT_SIZE-1 -: PADDING*BIT_SIZE] = 0;
        end
            assign in_act_padded[z*((IN_SIZE+(2*PADDING))**2)*BIT_SIZE + (PADDING+x)*(IN_SIZE+(2*PADDING))*BIT_SIZE + PADDING*BIT_SIZE +: IN_SIZE*BIT_SIZE] = reg_in_activations[(z*(IN_SIZE**2)*BIT_SIZE + x*IN_SIZE*BIT_SIZE) +: IN_SIZE*BIT_SIZE];
    end
end
endgenerate

// Generate all possible Convolution Windows
// For example: if the input has a size of 4x4x1 and kernel is 3x3x1
// Then for stride = 1, there will be 4 convolution windows of size 3x3x1
generate
genvar m,n,o,p;
for (m=0; m<IN_CHANNELS; m=m+1) begin
    for (n=0; n<(((IN_SIZE-KER_SIZE+(2*PADDING))/STRIDE)+1); n=n+1) begin
				for (o=0; o<((((IN_SIZE+(2*PADDING))-KER_SIZE)/STRIDE)+1); o=o+1) begin
						for (p=0; p<KER_SIZE; p=p+1) begin
								assign conv_wind [n*((((IN_SIZE)-KER_SIZE+(2*PADDING))/STRIDE)+1) + o][(((KER_SIZE**2)*IN_CHANNELS*BIT_SIZE)-1) - (m*((KER_SIZE**2)*BIT_SIZE) + p*KER_SIZE*BIT_SIZE) -: KER_SIZE*BIT_SIZE] = in_act_padded[((((IN_SIZE+(2*PADDING))**2)*IN_CHANNELS*BIT_SIZE)-1) - (m*((IN_SIZE+(2*PADDING))**2)*BIT_SIZE + p*(IN_SIZE+(2*PADDING))*BIT_SIZE + o*STRIDE*BIT_SIZE + n*STRIDE*(IN_SIZE+(2*PADDING))*BIT_SIZE) -: KER_SIZE*BIT_SIZE];
						end
				end
    end
end
endgenerate

// Convert CSR parameters to 2D.
// CSR stands for Compressed Sparse Row Format.
// Check the paper(https://arxiv.org/abs/2001.01955) for more details about CSR.
generate
genvar i;
		for (i=0; i<NON_ZERO_WEIGHTS; i=i+1) begin
				assign index_2d[i] = index[(NON_ZERO_WEIGHTS*INDEX_BIT_SIZE-1)-INDEX_BIT_SIZE*i -: INDEX_BIT_SIZE];
		end
		for (i=0; i<(KER_SIZE*IN_CHANNELS); i=i+1) begin
				assign r_pointer_2d[i] = r_pointer[R_POINTER_BIT_SIZE*i +: R_POINTER_BIT_SIZE];
		end
endgenerate

// Get index positions of non zero weight
integer j,k,l;
always @ (posedge clk) begin
if(!rst) begin
		weight_index <= 0;
		out_activations <= 0;
		out_weights <= 0;
		reg_in_activations <= 0;
		operations_count <= 0;
		start_pos <= 0;
		calc_index_pos <= 0;
		conv_act_index <= 0;
		weight_sel_index <= 0;
		setup_completed <= 0;
		vgm_setup_completed <= 0;
		for (j=0; j<NON_ZERO_WEIGHTS; j=j+1) begin
				act_index[j] <= 0;
		end
end
// Wait for cycle to recieve the input activations and weights
else if (!setup_completed) begin
		setup_completed <= 1;
end

// Calculate all the index positions of non zero weights so that
// correct input activation can be selected
else if (!calc_index_pos) begin
    for (j=0; j<IN_CHANNELS; j=j+1) begin
        for(k=0; k<KER_SIZE; k=k+1) begin
            for (l=0; l<r_pointer_2d[(KER_SIZE*IN_CHANNELS-1) - (k + KER_SIZE*j)]; l=l+1) begin
                if (l) start_pos = start_pos + index_2d[weight_index-1];
                act_index[weight_index] = (((KER_SIZE**2)*IN_CHANNELS*BIT_SIZE)-1) - (j*(KER_SIZE**2)*BIT_SIZE + k*KER_SIZE*BIT_SIZE + l*BIT_SIZE + index_2d[weight_index]*BIT_SIZE + start_pos*BIT_SIZE);
                weight_index = weight_index + 1;	 
						end
            start_pos = 0;
        end
    end
		calc_index_pos <= 1;
		weight_index <= 0;
end

// After all the above requirements are satisfied, 
// we can start the VGM operation.
else if (!vgm_setup_completed) begin
		reg_in_activations <= in_activations;
		vgm_setup_completed <= 1;
end
else begin
		// Select correct input activation from each convolution window generated earlier
		// based in the location of Non Zero Weights in the structured sparse filter
    for (j=0; j<(((IN_SIZE-KER_SIZE+(2*PADDING))/STRIDE)+1)**2; j=j+1) begin
				out_activations[j*BIT_SIZE +: BIT_SIZE] <= conv_wind[(((IN_SIZE-KER_SIZE+(2*PADDING))/STRIDE)+1)**2 - 1 - j][act_index[conv_act_index] -: BIT_SIZE];
        if( conv_act_index == NON_ZERO_WEIGHTS-1)
						conv_act_index <= 0;
				else
						conv_act_index <= conv_act_index + 1;
    end
		
		// Update output weights with non zero weights of the structured Sparse filter one by one in each clock cycle.
    for (j=0; j<FILTERS; j=j+1) begin
				out_weights[(FILTERS*BIT_SIZE-1) - (j*BIT_SIZE) -: BIT_SIZE] <= in_weights[(FILTERS*NON_ZERO_WEIGHTS*BIT_SIZE-1) - (j*NON_ZERO_WEIGHTS*BIT_SIZE + weight_sel_index*BIT_SIZE) -: BIT_SIZE];
				if( weight_sel_index == NON_ZERO_WEIGHTS-1)
						weight_sel_index <= 0;
				else
						weight_sel_index <= weight_sel_index + 1;
    end
		operations_count <= operations_count + 1;
end

// Take new input once the processing of previous input is done.
// It takes non zero weights number of cycle to process one input
if (operations_count == NON_ZERO_WEIGHTS-1) begin
		operations_count <= 0;
    reg_in_activations <= in_activations;
end



end

endmodule
