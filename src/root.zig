pub const ConvConfig = @import("ConvConfig.zig").ConvConfig;
pub const ConvLayout = @import("ConvConfig.zig").ConvLayout;
pub const ConvBackend = @import("ConvConfig.zig").ConvBackend;
pub const ComputeScratchSize = @import("ConvConfig.zig").ComputeScratchSize;

pub const Conv = @import("Conv.zig").Conv;

test {
    _ = @import("Conv_Direct_NCHW.zig");
    _ = @import("Conv_Direct_NHWC.zig");
    _ = @import("Conv_Im2Col_NCHW.zig");
    _ = @import("Conv_Im2Col_NHWC.zig");
}
