{
  inputs = {
    nixpkgs.url = "github:cachix/devenv-nixpkgs/rolling";
    devenv.url = "github:cachix/devenv";
  };

  nixConfig = {
    extra-trusted-public-keys = "devenv.cachix.org-1:w1cLUi8dv3hnoSPGAuibQv+f9TZLr6cv/Hm9XgU50cw=";
    extra-substituters = "https://devenv.cachix.org";
  };

  outputs =
    { self, nixpkgs, devenv, ... }@inputs:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in
    {
      devShells.${system}.default = devenv.lib.mkShell {
        inherit inputs pkgs;
        modules = [
          (
            { pkgs, lib, ... }: {
              packages = [
                pkgs.uv
                pkgs.git
                pkgs.pre-commit
                pkgs.zlib
              ];

              languages.python.enable = true;
              languages.python.package = pkgs.python311;

              env = {
                UV_NO_MANAGED_PYTHON = "true";
              };

              enterShell = ''
                uv sync --all-extras --locked
                source .venv/bin/activate
              '';
            }
          )
        ];
      };
    };
}
