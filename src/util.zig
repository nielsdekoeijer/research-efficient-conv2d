pub inline fn GetStride(comptime shape: anytype, comptime dim: usize) usize {
    var stride: usize = 1;
    inline for (dim + 1..shape.len) |i| {
        stride *= shape[i];
    }

    return stride;
}
