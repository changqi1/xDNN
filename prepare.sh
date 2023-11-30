#!/bin/bash

sudo dnf install -y gmp-devel mpfr-devel libmpc-devel zlib-devel gcc-c++

wget https://ftp.gnu.org/gnu/gcc/gcc-12.3.0/gcc-12.3.0.tar.xz
tar -xvf gcc-12.3.0.tar.xz
mkdir gcc-build
cd gcc-build
../gcc-12.3.0/configure --prefix=/usr/local/gcc-12.3.0 --enable-languages=c,c++ --disable-multilib
make -j $(nproc)
sudo make install

echo 'export PATH=/usr/local/gcc-12.3.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/gcc-12.3.0/lib64/:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
gcc --version

