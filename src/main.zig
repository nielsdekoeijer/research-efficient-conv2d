const std = @import("std");
const lib = @import("root.zig");

fn ProfileConv(comptime iterations: usize, comptime config: anytype) !void {
    const writer = std.io.getStdOut().writer();
    try writer.print("Running {any}::{any} {} times\n", .{ config.layout, config.backend, iterations });

    // scratch
    const alloc = std.heap.page_allocator;

    const len_inp0 = config.inp0_shape[0] * config.inp0_shape[1] *
        config.inp0_shape[2] * config.inp0_shape[3];
    const len_inp1 = config.inp1_shape[0] * config.inp1_shape[1] *
        config.inp1_shape[2] * config.inp1_shape[3];
    const len_out0 = config.out0_shape[0] * config.out0_shape[1] *
        config.out0_shape[2] * config.out0_shape[3];
    const len_scr0 = lib.ComputeScratchSize(config);

    const inp0: []align(config.A) config.T = try alloc.alignedAlloc(
        config.T,
        std.mem.Alignment.@"64",
        len_inp0,
    );
    defer alloc.free(inp0);

    const inp1: []align(config.A) config.T = try alloc.alignedAlloc(
        config.T,
        std.mem.Alignment.@"64",
        len_inp1,
    );
    defer alloc.free(inp1);

    const out0: []align(config.A) config.T = try alloc.alignedAlloc(
        config.T,
        std.mem.Alignment.@"64",
        len_out0,
    );
    defer alloc.free(out0);

    const scr0: []align(config.A) config.T = try alloc.alignedAlloc(
        config.T,
        std.mem.Alignment.@"64",
        len_scr0,
    );
    defer alloc.free(scr0);

    for (inp0) |*v| v.* = 0;
    for (inp1) |*v| v.* = 0;
    for (out0) |*v| v.* = 0;
    for (scr0) |*v| v.* = 0;

    var prng = std.Random.DefaultPrng.init(0xDEADBEEF);
    const rng = prng.random();

    var results = std.mem.zeroes([iterations]f64);
    for (0..iterations) |i| {
        for (inp0) |*v| v.* = rng.float(config.T) + v.*;
        for (inp1) |*v| v.* = rng.float(config.T) + v.*;
        for (out0) |*v| v.* = v.*;

        const start = std.time.nanoTimestamp();
        lib.Conv(config, inp0, inp1, out0, scr0);
        const end = std.time.nanoTimestamp();
        results[i] = @as(f64, @floatFromInt(end - start));
    }

    var acc: config.T = 0;
    for (out0) |v| {
        acc += v;
    }
    try writer.print("side effect: {any}\n", .{acc});

    var avg: f64 = 0.0;
    for (0..iterations) |i| {
        avg += results[i];
    }

    try writer.print("avg: {e:.3} ns\n", .{avg / iterations});
}

const builtin = @import("builtin");

pub fn main() !void {
    const iterations = 100;

    const scenarios = [_]struct {
        name: []const u8,
        iC: usize,
        oC: usize,
        iH: usize,
        iW: usize,
        k: usize,
        oH: usize,
        oW: usize,
    }{
        .{
            .name = "small",
            .iC = 4,
            .oC = 4,
            .iH = 8,
            .iW = 8,
            .k = 2,
            .oH = 9,
            .oW = 9,
        },
        .{
            .name = "medium",
            .iC = 16,
            .oC = 32,
            .iH = 32,
            .iW = 32,
            .k = 3,
            .oH = 32,
            .oW = 32,
        },
        // .{
        //     .name = "large",
        //     .iC = 16,
        //     .oC = 32,
        //     .iH = 32,
        //     .iW = 32,
        //     .k = 9,
        //     .oH = 26,
        //     .oW = 26,
        // },
    };

    const writer = std.io.getStdOut().writer();

    const features = @typeInfo(std.Target.x86.Feature).@"enum".fields;

    try writer.print(
        "Compiled for cpu: '{s}' '{s}', with features:\n",
        .{
            @tagName(builtin.cpu.arch),
            builtin.cpu.model.name,
        },
    );
    inline for (features) |field| {
        if (builtin.cpu.features.isEnabled(field.value)) {
            writer.print(" --> {s}\n", .{field.name}) catch {};
        }
    }

    inline for (scenarios) |scenario| {
        try writer.print("\nRunning '{s}'\n", .{scenario.name});

        try ProfileConv(iterations, lib.ConvConfig(f32, 64, lib.ConvLayout.NCHW, lib.ConvBackend.Direct){
            .inp0_shape = [_]usize{ 1, scenario.iC, scenario.iH, scenario.iW },
            .inp1_shape = [_]usize{ scenario.oC, scenario.iC, scenario.k, scenario.k },
            .out0_shape = [_]usize{ 1, scenario.oC, scenario.oH, scenario.oW },
            .kernel_shape = [_]usize{ scenario.k, scenario.k },
            .strides = [_]usize{ 1, 1 },
            .dilations = [_]usize{ 1, 1 },
            .pads = [_]usize{ 1, 1, 1, 1 },
        });

        try ProfileConv(iterations, lib.ConvConfig(f32, 64, lib.ConvLayout.NHWC, lib.ConvBackend.Direct){
            .inp0_shape = [_]usize{ 1, scenario.iH, scenario.iW, scenario.iC },
            .inp1_shape = [_]usize{ scenario.oC, scenario.k, scenario.k, scenario.iC },
            .out0_shape = [_]usize{ 1, scenario.oH, scenario.oW, scenario.oC },
            .kernel_shape = [_]usize{ scenario.k, scenario.k },
            .strides = [_]usize{ 1, 1 },
            .dilations = [_]usize{ 1, 1 },
            .pads = [_]usize{ 1, 1, 1, 1 },
        });

        try ProfileConv(iterations, lib.ConvConfig(f32, 64, lib.ConvLayout.NCHW, lib.ConvBackend.Im2Col){
            .inp0_shape = [_]usize{ 1, scenario.iC, scenario.iH, scenario.iW },
            .inp1_shape = [_]usize{ scenario.oC, scenario.iC, scenario.k, scenario.k },
            .out0_shape = [_]usize{ 1, scenario.oC, scenario.oH, scenario.oW },
            .kernel_shape = [_]usize{ scenario.k, scenario.k },
            .strides = [_]usize{ 1, 1 },
            .dilations = [_]usize{ 1, 1 },
            .pads = [_]usize{ 1, 1, 1, 1 },
        });

        try ProfileConv(iterations, lib.ConvConfig(f32, 8, lib.ConvLayout.NHWC, lib.ConvBackend.Im2Col){
            .inp0_shape = [_]usize{ 1, scenario.iH, scenario.iW, scenario.iC },
            .inp1_shape = [_]usize{ scenario.oC, scenario.k, scenario.k, scenario.iC },
            .out0_shape = [_]usize{ 1, scenario.oH, scenario.oW, scenario.oC },
            .kernel_shape = [_]usize{ scenario.k, scenario.k },
            .strides = [_]usize{ 1, 1 },
            .dilations = [_]usize{ 1, 1 },
            .pads = [_]usize{ 1, 1, 1, 1 },
        });
    }
}
