
module carrySelectAdder16bit(sum, in1, in2);
output [15:0] sum;
input [15:0] in1; 
input [15:0] in2;

//Intermideate wires for connection
wire c0,c1,c2,cout;

//Calculate 16 bit sum using 4 4-bit carry select adders.
carrySlelectAdder4bit cla0(sum[3:0], c0, in1[3:0], in2[3:0], 1'b0);
carrySlelectAdder4bit cla1(sum[7:4], c1, in1[7:4], in2[7:4], c0);
carrySlelectAdder4bit cla2(sum[11:8], c2, in1[11:8], in2[11:8], c1);
carrySlelectAdder4bit cla3(sum[15:12], cout, in1[15:12], in2[15:12], c2);

endmodule
