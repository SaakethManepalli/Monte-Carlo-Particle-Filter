`timescale 1ns/1ps
module lfsr_tb;
    reg  clk;
    reg  rst;
    wire [7:0] rand_val;

    integer fh;   // file handle
    integer i;    // loop counter

    // device under test
    lfsr8 dut (.clk(clk), .rst(rst), .out(rand_val));

    // 100 MHz clock
    always #5 clk = ~clk;

    initial begin
        clk = 0;   // initialize signals here, not at declaration
        rst = 1;

        // 1. open the file and write a header row
        fh = $fopen("results.csv", "w");
        $fdisplay(fh, "sample,value");   // column titles for the spreadsheet

        // release reset after a few cycles
        #20 rst = 0;

        // 2. write one row per sample
        for (i = 0; i < 1000; i = i + 1) begin
            @(posedge clk);
            $fdisplay(fh, "%0d,%0d", i, rand_val);   // "index,value"
        end

        // 3. close the file -- do not skip this
        $fclose(fh);
        $display("Wrote 1000 samples to results.csv");
        $finish;
    end
endmodule