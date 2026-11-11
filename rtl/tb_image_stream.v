// Testbench for image_stream. It checks the results by itself, so no waveform
// has to be looked at.
// A scoreboard remembers every item that goes in (in_valid && in_ready) and checks
// that every item coming out (out_valid && out_ready) is the same, in the same
// order, for both data and last. A second check makes sure the output does not
// change while it is stalled (out_valid && !out_ready).
`timescale 1ns/1ps

module tb_image_stream;
    localparam W = 8;

    reg              clk = 0;
    reg              rst = 1;
    reg  [W-1:0]     in_data = 0;
    reg              in_valid = 0;
    wire             in_ready;
    reg              in_last = 0;
    wire [W-1:0]     out_data;
    wire             out_valid;
    reg              out_ready = 0;
    wire             out_last;

    image_stream #(.DATA_WIDTH(W)) dut (
        .clk(clk), .rst(rst),
        .in_data(in_data), .in_valid(in_valid), .in_ready(in_ready), .in_last(in_last),
        .out_data(out_data), .out_valid(out_valid), .out_ready(out_ready), .out_last(out_last)
    );

    always #5 clk = ~clk;

    // ---------------- scoreboard ----------------
    // Lists of what went in (data and last), with a write and a read position
    reg [W-1:0] exp_data [0:4095];
    reg         exp_last [0:4095];
    integer wr = 0, rd = 0;
    integer errors = 0, in_xfers = 0, out_xfers = 0, last_seen = 0;

    // previous-cycle sample for stall-stability check
    reg           p_stall = 0;
    reg [W-1:0]   p_data;
    reg           p_last;

    // Runs every clock cycle: records inputs, checks outputs, checks stalls.
    // A reset empties the lists, because the DUT also throws its data away.
    always @(posedge clk) begin
        if (rst) begin
            wr <= 0; rd <= 0; p_stall <= 0;
        end else begin
            // input side
            if (in_valid && in_ready) begin
                exp_data[wr] = in_data;
                exp_last[wr] = in_last;
                wr = wr + 1;
                in_xfers = in_xfers + 1;
            end
            // output side
            if (out_valid && out_ready) begin
                if (rd >= wr) begin
                    $display("ERROR t=%0t: output transfer with nothing expected", $time);
                    errors = errors + 1;
                end else begin
                    if (out_data !== exp_data[rd] || out_last !== exp_last[rd]) begin
                        $display("ERROR t=%0t: got data=%h last=%b, expected data=%h last=%b",
                                 $time, out_data, out_last, exp_data[rd], exp_last[rd]);
                        errors = errors + 1;
                    end
                    if (out_last) last_seen = last_seen + 1;
                    rd = rd + 1;
                    out_xfers = out_xfers + 1;
                end
            end
            // stall stability: held output must not change while !out_ready
            if (p_stall && (!out_valid || out_data !== p_data || out_last !== p_last)) begin
                $display("ERROR t=%0t: output changed/dropped during stall", $time);
                errors = errors + 1;
            end
            p_stall <= out_valid && !out_ready;
            p_data  <= out_data;
            p_last  <= out_last;
        end
    end

    // ---------------- helpers (drive on negedge, sample on posedge) ----------------
    // send: put one item on the input and keep it there until the DUT accepts it
    task send;
        input [W-1:0] d;
        input         l;
        begin
            @(negedge clk);
            in_data = d; in_last = l; in_valid = 1;
            @(posedge clk);
            while (!in_ready) @(posedge clk);   // held until accepted
            #1;
        end
    endtask

    // idle: stop sending (valid goes low)
    task idle;
        begin
            @(negedge clk);
            in_valid = 0; in_last = 0;
        end
    endtask

    // drain: let the output run freely until everything that went in has come out
    task drain;
        integer guard;
        begin
            idle;
            out_ready = 1;
            guard = 0;
            while ((rd < wr || out_valid) && guard < 1000) begin
                @(posedge clk); guard = guard + 1;
            end
            @(posedge clk);
        end
    endtask

    // check: count an error and print its name if the condition is false
    task check;
        input [255:0] name;
        input integer cond;
        begin
            if (!cond) begin
                $display("ERROR: check failed: %0s", name);
                errors = errors + 1;
            end
        end
    endtask

    integer i;
    integer seed = 1;

    initial begin
        // ===== Reset behaviour =====
        $display("TEST reset");
        repeat (3) @(posedge clk);
        #1;
        check("out_valid low in reset", out_valid === 1'b0);
        check("out_last low in reset", out_last === 1'b0);
        @(negedge clk); rst = 0;
        out_ready = 1;

        // ===== Normal single transfer =====
        $display("TEST single transfer");
        send(8'hA5, 1'b0);
        idle;
        repeat (3) @(posedge clk);
        check("one item delivered", out_xfers == 1);

        // ===== Consecutive valid transfers (full throughput) =====
        $display("TEST consecutive transfers");
        for (i = 0; i < 8; i = i + 1) send(8'h10 + i, 1'b0);
        idle;
        drain;
        check("9 items out so far", out_xfers == 9);

        // ===== Frame end preserved =====
        $display("TEST frame end");
        for (i = 0; i < 4; i = i + 1) send(8'h20 + i, i == 3);
        drain;
        check("last flag seen once", last_seen == 1);

        // ===== Output stall / backpressure to the input =====
        $display("TEST output stall and input backpressure");
        out_ready = 0;
        send(8'h31, 1'b0);              // fills the register
        @(negedge clk);
        in_data = 8'h32; in_last = 1; in_valid = 1;   // second item must wait
        repeat (4) begin
            @(posedge clk); #1;
            check("in_ready low while stalled & full", in_ready === 1'b0);
            check("output holds first item", out_data === 8'h31 && out_valid === 1'b1);
        end
        @(negedge clk); out_ready = 1;  // release
        @(posedge clk);
        while (!in_ready) @(posedge clk);
        #1;
        idle;
        drain;
        check("both stalled items delivered in order", last_seen == 2);

        // ===== Reset mid-stream clears state =====
        $display("TEST reset mid-stream");
        out_ready = 0;
        send(8'h41, 1'b1);
        @(negedge clk); in_valid = 0;
        @(negedge clk); rst = 1;
        @(posedge clk); #1;
        check("out_valid cleared by reset", out_valid === 1'b0);
        check("out_last cleared by reset", out_last === 1'b0);
        @(negedge clk); rst = 0; out_ready = 1;
        repeat (3) begin
            @(posedge clk); #1;
            // a stale item would also trip the scoreboard ("nothing expected")
            check("no stale item after reset", out_valid === 1'b0);
        end

        // ===== Random traffic with random stalls =====
        $display("TEST random traffic");
        for (i = 0; i < 400; i = i + 1) begin
            out_ready = ($random(seed) & 3) != 0;          // ~75% ready
            if (($random(seed) & 1) == 0) begin
                @(negedge clk);
                in_data = $random(seed); in_last = (($random(seed) & 7) == 0);
                in_valid = 1;
                @(posedge clk);
                #1;
                if (!in_ready) begin
                    // hold until accepted, randomly toggling out_ready
                    while (!in_ready) begin
                        @(negedge clk); out_ready = ($random(seed) & 1);
                        @(posedge clk); #1;
                    end
                end
            end else begin
                idle;
                @(posedge clk);
            end
        end
        drain;

        check("scoreboard empty", rd == wr);
        $display("SUMMARY: %0d input transfers, %0d output transfers, %0d frames ends, %0d errors",
                 in_xfers, out_xfers, last_seen, errors);
        if (errors == 0) $display("PASS");
        else             $display("FAIL");
        $finish;
    end

    initial begin
        #2000000;
        $display("FAIL: timeout");
        $finish;
    end
endmodule
