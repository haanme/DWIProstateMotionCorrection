#!/bin/bash

sed -i '' "s:eija/Desktop:PET/Desktop/Harri:g" *.py
sed -i '' "s:eija/Desktop:PET/Desktop/Harri:g" runall*.sh
sed -i '' "s:eija/Documents/SW/Elastix/elastix_sources_v4.7/bin/bin:PET/Desktop/Harri/SW/Elastix/elastix_macosx64_v4.7/bin:g" *.py
sed -i '' "s:mcverter_basedir = '':mcverter_basedir = '/Users/PET/Desktop/Harri/SW/':g" conversions.py
