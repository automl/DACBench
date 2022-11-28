OS=$(uname)
if [ "$OS" == "Darwin" ]; then
    OS_NAME="osx"
    BIN_DIR="bin"
elif [ "$OS" == "Linux" ]; then
    OS_NAME="linux"
    BIN_DIR="bin64"
else
    echo "Sorry, only Linux and MacOS are supported. Your OS: ${OS}"
    exit 1
fi

dComplier="dmd.2.101.0.${OS_NAME}"
wget https://downloads.dlang.org/releases/2022/${dComplier}.tar.xz --no-check-certificate
tar -xf $dComplier.tar.xz
dmd2/${OS_NAME}/${BIN_DIR}/dmd -O -release ./calculatePolicy.d ./data_management.d ./eas.d

