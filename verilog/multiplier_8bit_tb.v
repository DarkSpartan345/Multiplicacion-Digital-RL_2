`timescale 1ns/1ps

module multiplier_tb;
    reg [1:0] A;
    reg [1:0] B;
    wire [3:0] P;

    multiplier dut(.A(A), .B(B), .P(P));

    initial begin
        // Caso 0: Prueba aleatoria 0
                A = 8'd3;
                B = 8'd1;
                #10;
                $display("%d", P);
// Caso 1: Prueba aleatoria 1
                A = 8'd3;
                B = 8'd3;
                #10;
                $display("%d", P);
// Caso 2: Prueba aleatoria 2
                A = 8'd2;
                B = 8'd1;
                #10;
                $display("%d", P);
// Caso 3: Prueba aleatoria 3
                A = 8'd2;
                B = 8'd1;
                #10;
                $display("%d", P);
// Caso 4: Prueba aleatoria 4
                A = 8'd3;
                B = 8'd1;
                #10;
                $display("%d", P);
// Caso 5: Prueba aleatoria 5
                A = 8'd1;
                B = 8'd1;
                #10;
                $display("%d", P);
// Caso 6: Prueba aleatoria 6
                A = 8'd3;
                B = 8'd1;
                #10;
                $display("%d", P);
// Caso 7: Prueba aleatoria 7
                A = 8'd2;
                B = 8'd3;
                #10;
                $display("%d", P);
// Caso 8: Prueba aleatoria 8
                A = 8'd2;
                B = 8'd3;
                #10;
                $display("%d", P);
// Caso 9: Prueba aleatoria 9
                A = 8'd1;
                B = 8'd2;
                #10;
                $display("%d", P);
// Caso 10: Prueba aleatoria 10
                A = 8'd1;
                B = 8'd2;
                #10;
                $display("%d", P);
// Caso 11: Prueba aleatoria 11
                A = 8'd1;
                B = 8'd2;
                #10;
                $display("%d", P);
// Caso 12: Prueba aleatoria 12
                A = 8'd2;
                B = 8'd2;
                #10;
                $display("%d", P);
// Caso 13: Prueba aleatoria 13
                A = 8'd1;
                B = 8'd3;
                #10;
                $display("%d", P);
// Caso 14: Prueba aleatoria 14
                A = 8'd3;
                B = 8'd3;
                #10;
                $display("%d", P);
// Caso 15: Prueba aleatoria 15
                A = 8'd2;
                B = 8'd1;
                #10;
                $display("%d", P);
        $finish;
    end

endmodule
