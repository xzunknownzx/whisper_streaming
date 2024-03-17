{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/master";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        python = pkgs.python311;
      in
      {
        packages = rec {
          default = python.pkgs.buildPythonApplication {
            pname = "stt";
            version = "0.1.0";
            src = ./.;
            format =
              "other"; # Specify that this is not a setuptools-based package
            propagatedBuildInputs = with python.pkgs; [
              faster-whisper
              numpy
              pydantic
              soundfile
              structlog
              websockets
            ];
            buildPhase = "true"; # otherwise errors
            installPhase = ''
              # Install the main executable
              mkdir -p $out/bin
              cp ws_server.py $out/bin/stt
              chmod +x $out/bin/stt
              # Install Python modules
              mkdir -p $out/${python.sitePackages}
              cp *.py $out/${python.sitePackages}
              # Create a wrapper script that sets PYTHONPATH
              makeWrapper $out/bin/stt $out/bin/stt-wrapper \
                --set PYTHONPATH $out/${python.sitePackages}
            '';
          };
          dockerImage = pkgs.dockerTools.buildImage {
            name = "stt";
            tag = "latest";
            copyToRoot = pkgs.buildEnv {
              name = "stt";
              paths = [ default ];
            };

            config = { Cmd = [ "${default}/bin/stt-wrapper" ]; };
          };
        };
        devShell = pkgs.mkShell {
          nativeBuildInputs = with pkgs;
            [
              (python.withPackages (ps:
                with ps; [
                  numpy
                  pydantic
                  soundfile
                  structlog
                  websockets
                  faster-whisper
                ]))
            ];
        };
      });
}
