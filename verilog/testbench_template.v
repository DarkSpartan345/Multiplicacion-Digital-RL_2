`timescale 1ns/1ps

module multiplier_tb;
    reg [{regsI}:0] A;
    reg [{regsI}:0] B;
    wire [{regsO}:0] P;

    multiplier dut(.A(A), .B(B), .P(P));

    initial begin
        {Test}
        $finish;
    end

endmodule
