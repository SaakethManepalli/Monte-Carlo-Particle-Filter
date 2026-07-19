`timescale 1ns/1ps

module lfsr8 (
    input            clk,
    input            rst,
    output reg [7:0] out
);
    // Maximal-length 8-bit LFSR
    // Feedback polynomial: x^8 + x^6 + x^5 + x^4 + 1
    // Taps at bits 8, 6, 5, 4 (1-indexed) = indices 7, 5, 4, 3 (0-indexed)
    // Period: 2^8 - 1 = 255 (cycles through all non-zero states)

    wire feedback = out[7] ^ out[5] ^ out[4] ^ out[3];

    always @(posedge clk) begin
        if (rst)
            out <= 8'hAC;                    // non-zero seed (any non-zero value works)
        else
            out <= {out[6:0], feedback};     // shift left, feed new bit into LSB
    end

endmodule