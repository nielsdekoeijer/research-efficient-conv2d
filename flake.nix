{
  description = "efficient-convd2";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    utils.url = "github:numtide/flake-utils";
    zig.url = "github:mitchellh/zig-overlay";
  };

  outputs = { self, nixpkgs, utils, zig }:
    utils.lib.eachSystem
      [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ]
      (
        system:
        let
          # packages for the given system
          pkgs = import nixpkgs {
            inherit system;
            overlays = [
              (final: prev: {
                zig = zig.packages.${system}."master";
              })
            ];
          };

        in
        {
          # on `nix build`
          packages.default = pkgs.callPackage ./build.nix {
            # openblas = openblas_x86;
            # target = "x86_64-linux-gnu";
          };

          # on `nix develop`
          devShells.default =
            let
              openblas_x86 = pkgs.openblas.override {
                enableStatic = true;
                singleThreaded = true;
                dynamicArch = true;
              };

              pkgs_aarch64 = import nixpkgs { crossSystem = { config = "aarch64-unknown-linux-musl"; }; };

              openblas_aarch64 = pkgs_aarch64.openblas.override {
                enableStatic = true;
                singleThreaded = true;
                target = "CORTEXA53";
              };
            in
            pkgs.mkShell {
              nativeBuildInputs = [
                pkgs.zig
                pkgs.bash
                pkgs.pkg-config
                pkgs.file
                pkgs.linuxKernel.packages.linux_6_13.perf
                pkgs.python312Packages.torch
              ];

              buildInputs = [
                openblas_x86
              ];

              shellHook = ''
                PS1="(dev) $PS1"
              '';
            };
        }
      );
}
