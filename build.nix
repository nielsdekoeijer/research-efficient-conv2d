{ lib
, zig
, stdenv
, openblas
, target ? "native"
}:
stdenv.mkDerivation {
  pname = "efficient-conv2d";
  version = "0.1.0";

  src = ./.;

  nativeBuildInputs = [ zig openblas ];

  buildInputs = [ openblas ];

  preBuild = ''
    export HOME=$TMPDIR
  '';

  installPhase = ''
     runHook preInstall
     zig build -Doptimize=ReleaseFast -Dtarget=${target} --prefix $out install
     runHook postInstall
   '';

  outputs = [ "out" ];

  meta = with lib; {
    description = "AWB parser";
    homepage = "https://github.com/bo-nemk/tech-efficient-conv2d";
    license = licenses.mit;
    platforms = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
  };
}
