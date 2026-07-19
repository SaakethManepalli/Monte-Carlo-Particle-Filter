//AI Generated, lfsr_tb is the one given to us by the winners dinner doc
// ─── tb_lfsr.v ─────────────────────────────────────────────────────────────
// Diffs lfsr_byte_gen output, byte-by-byte, against lfsr_golden.hex
// (exported directly from the Python/pylfsr reference model).
// ─────────────────────────────────────────────────────────────────────────
`timescale 1ns/1ps
 
module tb_lfsr;
 
    parameter N_VECTORS = 64;
 
    reg         clk;
    reg         rst_n;
    reg         start;
    wire [7:0]  byte_out;
    wire        valid;
 
    reg [7:0]   golden [0:N_VECTORS-1];
    integer     idx;
    integer     errors;
 
    lfsr_byte_gen #(.SEED(8'h11)) dut (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (start),
        .byte_out (byte_out),
        .valid    (valid)
    );
 
    initial clk = 1'b0;
    always #5 clk = ~clk;   // 100 MHz
 
    initial begin
        $readmemh("lfsr_golden.hex", golden);
 
        rst_n  = 1'b0;
        start  = 1'b0;
        idx    = 0;
        errors = 0;
        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);
 
        for (idx = 0; idx < N_VECTORS; idx = idx + 1) begin
            // pulse start for one cycle to kick off the next byte
            @(negedge clk);
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;
 
            // wait for valid (7 more cycles after the start-consuming cycle)
            wait (valid == 1'b1);
            if (byte_out !== golden[idx]) begin
                $display("MISMATCH at %0d: got %02h expected %02h", idx, byte_out, golden[idx]);
                errors = errors + 1;
            end else begin
                $display("OK       at %0d: got %02h expected %02h", idx, byte_out, golden[idx]);
            end
            @(posedge clk); // let valid deassert before next start
        end
 
        $display("---------------------------------------------");
        $display("Done. %0d/%0d mismatches", errors, N_VECTORS);
        if (errors == 0)
            $display("PASS: LFSR bit-accurate match to Python model.");
        else
            $display("FAIL: tap mapping or shift direction is wrong.");
        $finish;
    end
 
endmodule
