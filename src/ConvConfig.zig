pub const ConvLayout = enum {
    NHWC,
    NCHW,
};

pub const ConvBackend = enum {
    Direct,
    Im2Col,
};

pub fn ConvConfig(
    comptime T: type,
    comptime A: u32,
    comptime layout: ConvLayout,
    comptime backend: ConvBackend,
) type {
    return struct {
        comptime T: type = T,
        comptime A: u32 = A,
        comptime layout: ConvLayout = layout,
        comptime backend: ConvBackend = backend,
        kernel_shape: [2]usize,
        dilations: [2]usize,
        strides: [2]usize,
        pads: [4]usize,
        inp0_shape: [4]usize,
        inp1_shape: [4]usize,
        out0_shape: [4]usize,
    };
}

pub fn ComputeScratchSize(
    comptime config: anytype,
) usize {
    if (config.layout == ConvLayout.NHWC) {
        const oH = config.out0_shape[1];
        const oW = config.out0_shape[2];
        const kH = config.inp1_shape[1];
        const kW = config.inp1_shape[2];
        const iC = config.inp0_shape[3];

        return oH * oW * kH * kW * iC;
    }

    if (config.layout == ConvLayout.NCHW) {
        const oH = config.out0_shape[2];
        const oW = config.out0_shape[3];
        const kH = config.inp1_shape[2];
        const kW = config.inp1_shape[3];
        const iC = config.inp0_shape[1];

        return oH * oW * kH * kW * iC;
    }

    @compileError("not implemented");
}
