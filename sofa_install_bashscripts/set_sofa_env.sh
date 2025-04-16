#!/bin/bash
FOLDER_SRC=$HOME/sofa/src
FOLDER_TARGET=$HOME/sofa/build
FOLDER_SP3=$FOLDER_SRC/applications/plugins/SofaPython3
PYTHON_PKG_PATH=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
PYTHON_EXE=$(which python3)
PYTHON_ROOT_DIR=$CONDA_PREFIX

conda env config vars set SOFA_ROOT=$FOLDER_TARGET/install
conda env config vars set SOFAPYTHON3_ROOT=$FOLDER_TARGET/install/plugins/SofaPython3
