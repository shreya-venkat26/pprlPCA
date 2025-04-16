#!/bin/bash
conda create -n sofa-env -c conda-forge python=3.9 cmake=3.26 ninja qt=5 pybind11 \
    compilers libglvnd libglu glew libjpeg-turbo libpng libtiff eigen zlib pkg-config \
    boost \
    libpng \
    libjpeg-turbo \
    libtiff \
    glew \
    zlib \
    eigen \
    qt \
    cmake \
    ninja \
    pkg-config \
    gcc_linux-64 \
    gxx_linux-64
