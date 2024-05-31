////////////////////////////////////////////////////////////////////////////
////                  Main 16 bit MAC Module                            ////
////////////////////////////////////////////////////////////////////////////

module mac_16bit(out_rounded,md,mr,clk,rst);
parameter NON_ZERO_WEIGHTS = 27,
					FRACTIONAL_BITS = 8;

output reg [26:0] out_rounded;
input [15:0] md,mr;              //md: Multiplicand, mr: Multiplier
input clk,rst;

//Registor to store input multiplicand and multiplier
reg [15:0] MD,MR;

//Accumulator resgistor to store sum of all miltiplications
reg [39:0] acc;

//Wire coming out from multiplier
wire [39:0] mult_out;

//Wire coming out from partial product generator
wire [17:0] PP0,PP1,PP2,PP3,PP4,PP5,PP6,PP7,PP8;

//Registor to store sign extended 32 bit partial products
reg [31:0] PP0_reg,PP1_reg,PP2_reg,PP3_reg,PP4_reg,PP5_reg,PP6_reg,PP7_reg,PP8_reg;

//Wire of sign extended 32 bit partial product which will be assigned
//to above registors at every clock cycle
wire [31:0] PP0_FULL,PP1_FULL,PP2_FULL,PP3_FULL,PP4_FULL,PP5_FULL,PP6_FULL,PP7_FULL,PP8_FULL;

//Negate bit to get complete 2's complement when calculating partial products
wire [7:0] neg;

//Wire of sum and carry of all 7 Carry Save Adders used in wallace tree
wire [31:0] sum0,carry0,sum1,carry1,sum2,carry2,sum3,carry3,sum4,carry4,sum5,carry5,sum6,carry6;

//Wire for latest accumulator value
wire [39:0] new_acc;

//Counts clock cycle before the first output comes. Due to pipelining
//this needs to be done so that main counter can be started at 
//correct clock cycle 
reg [2:0] setup_count;

//Flag to denote set up of circuit is done to produce first output
reg setup_completed;

//Counter registor to count number of operations done
reg signed [$clog2(NON_ZERO_WEIGHTS)+1:0] count;

//Wires for final rounded down output
wire [26:0] out_rounded_wire;

//Generate 18 bit partial procucts and connect them to wires
generatePartialProducts PP(.PP0(PP0),.PP1(PP1),.PP2(PP2),.PP3(PP3),.PP4(PP4),.PP5(PP5),.PP6(PP6),.PP7(PP7),.PP8(PP8),.neg(neg),.MD(MD),.MR(MR));


//Sign extend 18 bit Partial products to 32 bit and also add negate bit wherever necessary
genvar i;

assign PP0_FULL[16:0] = PP0;
for(i=31; i>16; i=i-1) begin
  assign PP0_FULL[i] = PP0_FULL[16];
end

assign PP1_FULL[18:2] = PP1;
for(i=31; i>18; i=i-1) begin
  assign PP1_FULL[i] = PP1_FULL[18];
end
assign PP1_FULL[1] = 1'b0;
assign PP1_FULL[0] = neg[0];

assign PP2_FULL[20:4] = PP2;
for(i=31; i>20; i=i-1) begin
  assign PP2_FULL[i] = PP2_FULL[20];
end
assign PP2_FULL[3] = 1'b0;
assign PP2_FULL[2] = neg[1];
for(i=0; i<2; i=i+1) begin
  assign PP2_FULL[i] = 1'b0;
end

assign PP3_FULL[22:6] = PP3;
for(i=31; i>22; i=i-1) begin
  assign PP3_FULL[i] = PP3_FULL[22];
end
assign PP3_FULL[5] = 1'b0;
assign PP3_FULL[4] = neg[2];
for(i=0; i<4; i=i+1) begin
  assign PP3_FULL[i] = 1'b0;
end

assign PP4_FULL[24:8] = PP4;
for(i=31; i>24; i=i-1) begin
  assign PP4_FULL[i] = PP4_FULL[24];
end
assign PP4_FULL[7] = 1'b0;
assign PP4_FULL[6] = neg[3];
for(i=0; i<6; i=i+1) begin
  assign PP4_FULL[i] = 1'b0;
end

assign PP5_FULL[26:10] = PP5;
for(i=31; i>26; i=i-1) begin
  assign PP5_FULL[i] = PP5_FULL[26];
end
assign PP5_FULL[9] = 1'b0;
assign PP5_FULL[8] = neg[4];
for(i=0; i<8; i=i+1) begin
  assign PP5_FULL[i] = 1'b0;
end

assign PP6_FULL[28:12] = PP6;
for(i=31; i>28; i=i-1) begin
  assign PP6_FULL[i] = PP6_FULL[28];
end
assign PP6_FULL[11] = 1'b0;
assign PP6_FULL[10] = neg[5];
for(i=0; i<10; i=i+1) begin
  assign PP6_FULL[i] = 1'b0;
end

assign PP7_FULL[30:14] = PP7;
assign PP7_FULL[31] = PP7_FULL[30];
assign PP7_FULL[13] = 1'b0;
assign PP7_FULL[12] = neg[6];
for(i=0; i<12; i=i+1) begin
  assign PP7_FULL[i] = 1'b0;
end

assign PP8_FULL[31:16] = PP8[15:0];
assign PP8_FULL[15] = 1'b0;
assign PP8_FULL[14] = neg[7];
for(i=0; i<14; i=i+1) begin
  assign PP8_FULL[i] = 1'b0;
end

//Wallace tree of 7 Carry Save Adders to get final sum and carry
carrySaveAdder32bit csa0(.sum(sum0),.carry(carry0),.in1(PP0_reg),.in2(PP1_reg),.in3(PP2_reg));
carrySaveAdder32bit csa1(.sum(sum1),.carry(carry1),.in1(PP3_reg),.in2(PP4_reg),.in3(PP5_reg));
carrySaveAdder32bit csa2(.sum(sum2),.carry(carry2),.in1(PP6_reg),.in2(PP7_reg),.in3(PP8_reg));
carrySaveAdder32bit csa3(.sum(sum3),.carry(carry3),.in1(sum0),.in2(carry0),.in3(sum1));
carrySaveAdder32bit csa4(.sum(sum4),.carry(carry4),.in1(carry1),.in2(sum2),.in3(carry2));
carrySaveAdder32bit csa5(.sum(sum5),.carry(carry5),.in1(sum3),.in2(carry3),.in3(sum4));
carrySaveAdder32bit csa6(.sum(sum6),.carry(carry6),.in1(sum5),.in2(carry5),.in3(carry4));

//Add final sum and carry to get booth encoded wallace tree multiplier output
carrySelectAdder32bit cla0(.sum(mult_out[31:0]), .in1(sum6), .in2(carry6));

//Sign extend 32 bit multiplier output to 40 bit to make it ready for adding in accumulator
for(i=32; i<40; i=i+1) begin
  assign mult_out[i] = mult_out[31];
end

//Add new multiplier output to accumulator registor's previous value
carrySelectAdder40bit cla1(.sum(new_acc), .in1(mult_out), .in2(acc));

// Assign rounded down output.
// FRACTIONAL_BITS parameter is used to assign correct bits
// as per the fractional bits we have used to store the weights and biases
assign out_rounded_wire = acc[FRACTIONAL_BITS + 26 : FRACTIONAL_BITS];

//Always block to run on every +ve clock edge
always @ (posedge clk) begin

		//If MAC is ready to produce first output, start count.
		if(setup_count == 3 && !setup_completed && rst) begin
				setup_completed <= 1;
				MD <= md;
				MR <= mr;
				PP0_reg <= PP0_FULL;
				PP1_reg <= PP1_FULL;
				PP2_reg <= PP2_FULL;
				PP3_reg <= PP3_FULL;
				PP4_reg <= PP4_FULL;
				PP5_reg <= PP5_FULL;
				PP6_reg <= PP6_FULL;
				PP7_reg <= PP7_FULL;
				PP8_reg <= PP8_FULL;
				acc <= new_acc;
				out_rounded <= out_rounded_wire;
				count <= 1;
		end

		//If setup is not yet completed keep passing data to next registors
		else if(!setup_completed && rst) begin
				MD <= md;
				MR <= mr;
				PP0_reg <= PP0_FULL;
				PP1_reg <= PP1_FULL;
				PP2_reg <= PP2_FULL;
				PP3_reg <= PP3_FULL;
				PP4_reg <= PP4_FULL;
				PP5_reg <= PP5_FULL;
				PP6_reg <= PP6_FULL;
				PP7_reg <= PP7_FULL;
				PP8_reg <= PP8_FULL;
				acc <= new_acc;
				out_rounded <= out_rounded_wire;
				setup_count <= setup_count + 1;
		end

		//If main counter has started and no reset signal is given (Reset is active low).
		if(count >0 && rst) begin
				MD <= md;
				MR <= mr;
				PP0_reg <= PP0_FULL;
				PP1_reg <= PP1_FULL;
				PP2_reg <= PP2_FULL;
				PP3_reg <= PP3_FULL;
				PP4_reg <= PP4_FULL;
				PP5_reg <= PP5_FULL;
				PP6_reg <= PP6_FULL;
				PP7_reg <= PP7_FULL;
				PP8_reg <= PP8_FULL;
				if (count == NON_ZERO_WEIGHTS) begin
						acc <= mult_out;
						count <= 1;
				end
				else begin
						acc <= new_acc;
						count <= count + 1;
				end
				out_rounded <= out_rounded_wire;
		end

		//If reset signal is recieved. Reset all registors. Reset is active low.
		if(!rst) begin
				acc <= 0;
				MD <= 0;
				MR <= 0;
				out_rounded <= 0;
				setup_count <= 0;
				setup_completed <= 0;
				count <= 0;
				PP0_reg <= 0;
				PP1_reg <= 0;
				PP2_reg <= 0;
				PP3_reg <= 0;
				PP4_reg <= 0;
				PP5_reg <= 0;
				PP6_reg <= 0;
				PP7_reg <= 0;
				PP8_reg <= 0;
		end

		//If all 256 operations are completed, reset all registors to 0.
		if(count == 256) begin
				acc <= 0;
				out_rounded <= 0;
				setup_count <= 1;
				setup_completed <= 0;
				count <= 0;
				PP0_reg <= 0;
				PP1_reg <= 0;
				PP2_reg <= 0;
				PP3_reg <= 0;
				PP4_reg <= 0;
				PP5_reg <= 0;
				PP6_reg <= 0;
				PP7_reg <= 0;
				PP8_reg <= 0;
		end
end
endmodule

