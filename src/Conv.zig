const GetStride = @import("util.zig").GetStride;
const ConvConfig = @import("ConvConfig.zig").ConvConfig;
const ConvLayout = @import("ConvConfig.zig").ConvLayout;
const ConvBackend = @import("ConvConfig.zig").ConvBackend;
const Conv_Direct_NCHW = @import("Conv_Direct_NCHW.zig").Conv_Direct_NCHW;
const Conv_Im2Col_NCHW = @import("Conv_Im2Col_NCHW.zig").Conv_Im2Col_NCHW;
const Conv_Direct_NHWC = @import("Conv_Direct_NHWC.zig").Conv_Direct_NHWC;
const Conv_Im2Col_NHWC = @import("Conv_Im2Col_NHWC.zig").Conv_Im2Col_NHWC;

pub noinline fn Conv(
    comptime config: anytype,
    noalias inp0: []align(config.A) const config.T,
    noalias inp1: []align(config.A) const config.T,
    noalias out0: []align(config.A) config.T,
    noalias scr0: []align(config.A) config.T,
) void {
    if (config.layout == ConvLayout.NCHW and config.backend == ConvBackend.Direct) {
        return Conv_Direct_NCHW(config, inp0, inp1, out0);
    }
    if (config.layout == ConvLayout.NHWC and config.backend == ConvBackend.Direct) {
        return Conv_Direct_NHWC(config, inp0, inp1, out0);
    }
    if (config.layout == ConvLayout.NCHW and config.backend == ConvBackend.Im2Col) {
        return Conv_Im2Col_NCHW(config, inp0, inp1, out0, scr0);
    }
    if (config.layout == ConvLayout.NHWC and config.backend == ConvBackend.Im2Col) {
        return Conv_Im2Col_NHWC(config, inp0, inp1, out0, scr0);
    }

    @compileError("No support for this configuration");
}
