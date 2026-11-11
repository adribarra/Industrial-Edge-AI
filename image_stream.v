`timescale 1ns/1ps
// image_stream: a one-stage register that passes a stream of data along with a
// valid/ready handshake.
//
// This is a learning exercise for Phase 1. It is NOT connected to the AI Suite or
// to the Python application.
//
// How the handshake works: data moves on a clock edge only when valid and ready
// are both high.
//   in_ready = !out_valid || out_ready
//     The register accepts a new item when it is empty, or when the item it holds
//     is leaving on this same clock edge. That allows one transfer every cycle.
//     If the output is not ready, in_ready goes low so nothing gets overwritten
//     or lost, and the output keeps its data and last stable while it waits.
//   in_last marks the final item of a frame and travels together with its data.
//   Reset (synchronous, active high) empties the register, so no data or last flag
//   survives it.
module image_stream #(
    parameter DATA_WIDTH = 8
) (
    input  wire                  clk,
    input  wire                  rst,        // synchronous, active high

    input  wire [DATA_WIDTH-1:0] in_data,
    input  wire                  in_valid,
    output wire                  in_ready,
    input  wire                  in_last,

    output reg  [DATA_WIDTH-1:0] out_data,
    output reg                   out_valid,
    input  wire                  out_ready,
    output reg                   out_last
);

    // Ready for new data if the register is empty or is being emptied right now
    assign in_ready = !out_valid || out_ready;

    // The register itself: one item of data (plus its last flag) and a valid bit
    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0;
            out_data  <= {DATA_WIDTH{1'b0}};
            out_last  <= 1'b0;
        end else if (in_valid && in_ready) begin
            out_valid <= 1'b1;
            out_data  <= in_data;
            out_last  <= in_last;
        end else if (out_ready) begin
            out_valid <= 1'b0;          // item consumed, nothing new arrived
        end
    end

endmodule
