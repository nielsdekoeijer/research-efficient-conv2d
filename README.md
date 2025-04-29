# research-efficient-conv2d
Little repo with some research! WIP / incomplete.

```bash
# I do:
nix develop
zig build run -Doptimize=ReleaseFast -Dcpu=skylake
```

Yields:
```bash
Running 'small'
Running ConvConfig.ConvLayout.NCHW::ConvConfig.ConvBackend.Direct 100 times
side effect: 1.0440112e7
avg: 1.988e3 ns
Running ConvConfig.ConvLayout.NHWC::ConvConfig.ConvBackend.Direct 100 times
side effect: 1.0439851e7
avg: 8.870e2 ns
Running ConvConfig.ConvLayout.NCHW::ConvConfig.ConvBackend.Im2Col 100 times
side effect: 1.0440113e7
avg: 9.560e2 ns
Running ConvConfig.ConvLayout.NHWC::ConvConfig.ConvBackend.Im2Col 100 times
side effect: 1.0439851e7
avg: 8.185e2 ns

Running 'medium'
Running ConvConfig.ConvLayout.NCHW::ConvConfig.ConvBackend.Direct 100 times
side effect: 1.1305529e10
avg: 3.488e6 ns
Running ConvConfig.ConvLayout.NHWC::ConvConfig.ConvBackend.Direct 100 times
side effect: 1.1305774e10
avg: 9.796e5 ns
Running ConvConfig.ConvLayout.NCHW::ConvConfig.ConvBackend.Im2Col 100 times
side effect: 1.1305529e10
avg: 2.402e5 ns
Running ConvConfig.ConvLayout.NHWC::ConvConfig.ConvBackend.Im2Col 100 times
side effect: 1.1305774e10
avg: 1.155e5 ns
```
