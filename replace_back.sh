#!/bin/bash

sed -i '' "s:PET/Desktop/Harri:eija/Desktop:g" *.py
sed -i '' "s:PET/Desktop/Harri:eija/Desktop:g" runall*.sh
sed -i '' "s:PET/Desktop/Harri/SW/Elastix/elastix_macosx64_v4.7/bin:eija/Documents/SW/Elastix/elastix_sources_v4.7/bin/bin:g" *.py
