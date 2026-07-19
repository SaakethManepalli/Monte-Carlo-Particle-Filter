module lfsr_core #(
    parameter [7:0] SEED = 8'h11
)(
    input  wire clk,
    input  wire rst_n,
    input  wire enable,     // advance one bit per cycle when high
    output wire outbit      // bit shifted out THIS cycle (pre-shift LSB)
);

    reg [7:0] shift_reg;
    wire      feedback;

    // taps: positions 8,6,5,4 (pylfsr, 1-indexed) -> bits 0,2,3,4 (verilog)
    assign feedback = shift_reg[0] ^ shift_reg[2] ^ shift_reg[3] ^ shift_reg[4];
    assign outbit   = shift_reg[0];   // captured before the shift below

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            shift_reg <= SEED;
        else if (enable)
            shift_reg <= {feedback, shift_reg[7:1]};
    end

endmodule