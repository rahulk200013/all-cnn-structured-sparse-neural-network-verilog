module lut_r_pointer_1(sbyte,addr);
input [3:0] addr;
output reg [8:0] sbyte;

always @ (addr) begin

(* synthesis, full_case, parallel_case *) case (addr)

4'b0001: sbyte = 9'b011011011;
endcase
end
endmodule



module lut_r_pointer_2(sbyte,addr);
input [3:0] addr;
output reg [107:0] sbyte;

always @ (addr) begin

(* synthesis, full_case, parallel_case *) case (addr)

4'b0010: sbyte = 108'b011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011;
endcase
end
endmodule



module lut_r_pointer_3(sbyte,addr);
input [3:0] addr;
output reg [107:0] sbyte;

always @ (addr) begin

(* synthesis, full_case, parallel_case *) case (addr)

4'b0011: sbyte = 108'b011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011;
endcase
end
endmodule



module lut_r_pointer_4(sbyte,addr);
input [3:0] addr;
output reg [107:0] sbyte;

always @ (addr) begin

(* synthesis, full_case, parallel_case *) case (addr)

4'b0100: sbyte = 108'b011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011;
endcase
end
endmodule



module lut_r_pointer_5(sbyte,addr);
input [3:0] addr;
output reg [215:0] sbyte;

always @ (addr) begin

(* synthesis, full_case, parallel_case *) case (addr)

4'b0101: sbyte = 216'b011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011;
endcase
end
endmodule



module lut_r_pointer_6(sbyte,addr);
input [3:0] addr;
output reg [215:0] sbyte;

always @ (addr) begin

(* synthesis, full_case, parallel_case *) case (addr)

4'b0110: sbyte = 216'b011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011;
endcase
end
endmodule



module lut_r_pointer_7(sbyte,addr);
input [3:0] addr;
output reg [215:0] sbyte;

always @ (addr) begin

(* synthesis, full_case, parallel_case *) case (addr)

4'b0111: sbyte = 216'b011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011011;
endcase
end
endmodule



module lut_r_pointer_8(sbyte,addr);
input [3:0] addr;
output reg [107:0] sbyte;

always @ (addr) begin

(* synthesis, full_case, parallel_case *) case (addr)

4'b1000: sbyte = 108'b001001001001001001001001001001001001001001001001001001001001001001001001001001001001001001001001001001001001;
endcase
end
endmodule



module lut_r_pointer_9(sbyte,addr);
input [3:0] addr;
output reg [107:0] sbyte;

always @ (addr) begin

(* synthesis, full_case, parallel_case *) case (addr)

4'b1001: sbyte = 108'b001001001001001001001001001001001001001001001001001001001001001001001001001001001001001001001001001001001001;
endcase
end
endmodule



