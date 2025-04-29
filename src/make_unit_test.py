import torch

class setup:

    def __init__(self, ishape, kshape, stride, padding, dilation, groups):
        self.iN, self.iC, self.iH, self.iW = ishape
        self.kM, self.kC, self.kH, self.kW = kshape
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.inp = torch.randn(self.iN, self.iC, self.iH, self.iW)
        self.wgt = torch.randn(self.kM, self.kC, self.kH, self.kW)
        self.out = torch.nn.functional.conv2d(self.inp, self.wgt, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        self.oN, self.oC, self.oH, self.oW = self.out.shape


    def print(self, name, layout="NCHW", backend="Direct"):
        if (layout == "NCHW"):
            print(f"test \"{name}\" {{")
            print(
            f"""
        const cfg = ConvConfig(f32, 16, ConvLayout.{layout}, ConvBackend.Direct){{
            .inp0_shape = .{{ {self.iN}, {self.iC}, {self.iH}, {self.iW} }},
            .inp1_shape = .{{ {self.kM}, {self.kC}, {self.kH}, {self.kW} }},
            .out0_shape = .{{ {self.oN}, {self.oC}, {self.oH}, {self.oW} }},
            .kernel_shape = .{{ {self.kH}, {self.kW} }},
            .strides = .{{ {self.stride[0]}, {self.stride[1]} }},
            .dilations = .{{ {self.dilation[0]}, {self.dilation[1]} }},
            .pads = .{{ {self.padding[0]}, {self.padding[0]}, {self.padding[1]}, {self.padding[1]} }},
        }};
            """)

            print(f"    const inp: [{len(self.inp.flatten())}]f32 align(16)= .{{")
            for value in self.inp.flatten():
                print(f"        {value},")
            print(f"    }};")
            print(f"")

            print(f"    const wgt: [{len(self.wgt.flatten())}]f32 align(16)= .{{")
            for value in self.wgt.flatten():
                print(f"        {value},")
            print(f"    }};")
            print(f"")

            print(f"    const ref: [{len(self.out.flatten())}]f32 align(16)= .{{")
            for value in self.out.flatten():
                print(f"        {value},")
            print(f"    }};")
            print(f"")

            if backend=="Direct":
                print(f"    var out: [{len(self.out.flatten())}]f32 align(16)= .{{0}} ** {len(self.out.flatten())};")
                print(f"    Conv_Direct_{layout}(cfg, &inp, &wgt, &out);")

            if backend=="Im2Col":
                print(f"    var out: [{len(self.out.flatten())}]f32 align(16)= .{{0}} ** {len(self.out.flatten())};")
                scr = self.iC * self.kH * self.kW * self.oH * self.oW;
                print(f"    var scr: [{scr}]f32 align(16)= .{{0}} ** {scr};")
                print(f"    Conv_Im2Col_{layout}(cfg, &inp, &wgt, &out, &scr);")



            print(f"""
        for (out, 0..) |v, i| {{
            try testing.expect(std.math.approxEqAbs(f32, v, ref[i], 1e-4));
        }}
        """)
            print("}")
        if (layout == "NHWC"):
            print(f"test \"{name}\" {{")
            print(
            f"""
        const cfg = ConvConfig(f32, 16, ConvLayout.{layout}, ConvBackend.Direct){{
            .inp0_shape = .{{ {self.iN}, {self.iH}, {self.iW}, {self.iC} }},
            .inp1_shape = .{{ {self.kM}, {self.kH}, {self.kW}, {self.kC} }},
            .out0_shape = .{{ {self.oN}, {self.oH}, {self.oW}, {self.oC} }},
            .kernel_shape = .{{ {self.kH}, {self.kW} }},
            .strides = .{{ {self.stride[0]}, {self.stride[1]} }},
            .dilations = .{{ {self.dilation[0]}, {self.dilation[1]} }},
            .pads = .{{ {self.padding[0]}, {self.padding[0]}, {self.padding[1]}, {self.padding[1]} }},
        }};
            """)

            print(f"    const inp: [{len(self.inp.permute(0, 2, 3, 1).flatten())}]f32 align(16)= .{{")
            for value in self.inp.permute(0, 2, 3, 1).flatten():
                print(f"        {value},")
            print(f"    }};")
            print(f"")

            print(f"    const wgt: [{len(self.wgt.permute(0, 2, 3, 1).flatten())}]f32 align(16)= .{{")
            for value in self.wgt.permute(0, 2, 3, 1).flatten():
                print(f"        {value},")
            print(f"    }};")
            print(f"")

            print(f"    const ref: [{len(self.out.permute(0, 2, 3, 1).flatten())}]f32 align(16)= .{{")
            for value in self.out.permute(0, 2, 3, 1).flatten():
                print(f"        {value},")
            print(f"    }};")
            print(f"")

            if backend=="Direct":
                print(f"    var out: [{len(self.out.permute(0, 2, 3, 1).flatten())}]f32 align(16)= .{{0}} ** {len(self.out.permute(0, 2, 3, 1).flatten())};")
                print(f"    Conv_Direct_{layout}(cfg, &inp, &wgt, &out);")

            if backend=="Im2Col":
                print(f"    var out: [{len(self.out.permute(0, 2, 3, 1).flatten())}]f32 align(16)= .{{0}} ** {len(self.out.permute(0, 2, 3, 1).flatten())};")
                scr = self.iC * self.kH * self.kW * self.oH * self.oW;
                print(f"    var scr: [{scr}]f32 align(16)= .{{0}} ** {scr};")
                print(f"    Conv_Im2Col_{layout}(cfg, &inp, &wgt, &out, &scr);")

            print(f"""
        for (out, 0..) |v, i| {{
            try testing.expect(std.math.approxEqAbs(f32, v, ref[i], 1e-4));
        }}
        """)
            print("}")

setup(ishape=(1,1,5,3), kshape=(1,1,2,3), stride=(1,1), padding=(0,0), dilation=(1,1), groups=1).print("NHWC Im2Col test 1", backend="Im2Col", layout="NHWC")
print("")
setup(ishape=(2,4,3,5), kshape=(2,4,3,2), stride=(1,1), padding=(0,0), dilation=(1,1), groups=1).print("NHWC Im2Col test 2", backend="Im2Col", layout="NHWC")
print("")
setup(ishape=(2,4,3,5), kshape=(2,4,3,2), stride=(3,4), padding=(0,0), dilation=(1,1), groups=1).print("NHWC Im2Col test 3", backend="Im2Col", layout="NHWC")
print("")
setup(ishape=(2,4,3,5), kshape=(2,4,2,2), stride=(6,2), padding=(3,2), dilation=(1,1), groups=1).print("NHWC Im2Col test 4", backend="Im2Col", layout="NHWC")
print("")
setup(ishape=(2,4,3,5), kshape=(2,4,2,2), stride=(6,2), padding=(1,1), dilation=(1,2), groups=1).print("NHWC Im2Col test 5", backend="Im2Col", layout="NHWC")
