dComplier="dmd.2.100.0.linux"
wget https://s3.us-west-2.amazonaws.com/downloads.dlang.org/releases/2022/$dComplier.tar.xz
tar -xf $dComplier.tar.xz
dmd2/linux/bin64/dmd -O -release ./calculatePolicy.d ./data_management.d ./eas.d

