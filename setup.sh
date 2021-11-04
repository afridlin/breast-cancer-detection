#!/bin/sh

# run this script once before you start developing

# install required python modules
pip3 install numpy pandas matplotlib seaborn

# clone the git repository including some basic code and the breast cancer data
git clone https://github.com/afridlin/breast-cancer-detection.git

# go into the cloned folder
cd breast-cancer-detection

# open Visual Studio Code
code .