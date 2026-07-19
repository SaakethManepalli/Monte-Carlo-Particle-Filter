module lfsr_byte_gen #(
    parameter [7:0] SEED = 8'h11
)(
    input  wire       clk,
    input  wire       rst_n,
    input  wire       start,      // pulse to begin assembling next byte
    output reg  [7:0] byte_out,
    output reg         valid       // 1-cycle pulse when byte_out is complete
);

    wire       core_enable;
    wire       core_outbit;
    reg  [2:0] bit_cnt;      // counts 0..7
    reg        running;

    lfsr_core #(.SEED(SEED)) u_core (
        .clk     (clk),
        .rst_n   (rst_n),
        .enable  (core_enable),
        .outbit  (core_outbit)
    );

    assign core_enable = running;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            bit_cnt  <= 3'd0;
            byte_out <= 8'd0;
            running  <= 1'b0;
            valid    <= 1'b0;
        end else begin
            valid <= 1'b0; // default: only asserted the cycle a byte completes

            if (start && !running) begin
                running <= 1'b1;
                bit_cnt <= 3'd0;
            end else if (running) begin
                // val = (val << 1) | outbit  -- same order as the Python loop
                byte_out <= {byte_out[6:0], core_outbit};
                if (bit_cnt == 3'd7) begin
                    running <= 1'b0;
                    valid   <= 1'b1;
                end
                bit_cnt <= bit_cnt + 3'd1;
            end
        end
    end

endmodule