{
  clangStdenv,
  cmake,
  lib,
  ninja,
}:
let
  stdenv = clangStdenv;
  targets = [
    "01-texture-inference"
    "02-texture-training"
    "unittest"
  ];
in
stdenv.mkDerivation {
  pname = "minidxnn";
  version = "0.4.0";

  src = ../.;

  nativeBuildInputs = [
    cmake
    ninja
  ];

  cmakeFlags = [
    "-DMINIDXNN_BUILD_CPP_FALLBACK_ONLY=ON"
    "-DMINIDXNN_BUILD_TESTS=ON"
  ];

  buildPhase = ''
    cmake --build . --target ${lib.concatStringsSep " " targets}
  '';

  installPhase = ''
    mkdir -p $out/bin
    cp example/01-texture-inference example/02-texture-training unittest/unittest $out/bin/
  '';

  meta = with lib; {
    description = "A minimal DirectX-based neural network library";
    homepage = "https://github.com/GPUOpen-LibrariesAndSDKs/MiniDXNN";
    license = licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
