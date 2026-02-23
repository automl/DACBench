{
  description = "Develop Python on Nix with uv";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, ... }:
    let
      inherit (nixpkgs) lib;
      forAllSystems = lib.genAttrs lib.systems.flakeExposed;
    in
    {
      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          makeNixLDWrapper = program: (pkgs.runCommand "${program.pname}-nix-ld-wrapped" { } ''
            mkdir -p $out/bin
            for file in ${program}/bin/*; do
              new_file=$out/bin/$(basename $file)
              echo "#! ${pkgs.bash}/bin/bash -e" >> $new_file
              echo 'export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$NIX_LD_LIBRARY_PATH"' >> $new_file
              echo 'exec -a "$0" '$file' "$@"' >> $new_file
              chmod +x $new_file
            done
          '');
        in
        {
          default = pkgs.mkShell {
            packages = [
              (makeNixLDWrapper pkgs.python310)
              pkgs.uv
            ];

            env = lib.optionalAttrs pkgs.stdenv.isLinux rec {
              # Python libraries often load native shared objects using dlopen(3).
              # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
              # We do this here only for nix-ld, but there is a wrapper script above which wraps our python program to use it
              NIX_LD_LIBRARY_PATH = ''
              ${lib.makeLibraryPath [
                pkgs.stdenv.cc.cc.lib
              ]}:${lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1}
              '';
              NIX_LD = lib.fileContents "${pkgs.stdenv.cc}/nix-support/dynamic-linker";
            };

            shellHook = ''
              unset PYTHONPATH
              uv sync
              . .venv/bin/activate
            '';
          };
        }
      );
    };
}
