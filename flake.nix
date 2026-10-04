{
  description = "MiniDXNN devenv";

  inputs = {
    self.submodules = true;
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    {
      nixpkgs,
      ...
    }:
    let
      lib = nixpkgs.lib;

      systems = [
        "x86_64-linux"
      ];

      systemAttrs = f: system: { ${system} = f system; };

      mapSystems = f: systems |> lib.map (systemAttrs f) |> lib.mergeAttrsList;

      packages = mapSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = pkgs.callPackage ./nix { };
        }
      );

      devShells = mapSystems (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          default = pkgs.mkShell.override { stdenv = pkgs.clangStdenv; } {
            inputsFrom = [ packages.${system}.default ];
            packages = with pkgs; [
              clang-tools
              cmake-language-server
              shader-slang
            ];
          };
        }
      );
    in
    {
      inherit packages devShells;
    };
}
