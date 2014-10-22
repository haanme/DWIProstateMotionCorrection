#!/bin/sh

#  Script.sh
#  
#
#  Created by Harri Merisaari on 3/10/14.
#
set +e
binaryname="python motioncorrection_2nd.py"
binaryname2="python motioncorrection_transformix.py"
param_rigid="Par0001translation_mod_ver4.txt"
param_BSpline="Par0001bspline08_mod_ver4.txt"
no_folders=$(ls -l -d ../PET_MR_dwis/* | grep '^d' | grep -v 'Carimas27' | grep -v 'ProjectTemplate' | awk '{print $9}' | wc -l)
folders=$(ls -l -d ../PET_MR_dwis/* | grep '^d' | grep -v 'Carimas27' | grep -v 'ProjectTemplate' | awk '{print $9}')
# start
echo $no_folders " folders to be processed"
let "round = 1"
for f in $folders
do
    subject_id=$(echo $f | awk -F '/' '{print $3}')

#    subject_id=$(echo "10_moring_hB_1a")
#    f=$(echo "../PET_MR_dwis/10_moring_hB_1a")

    xtermcmd=$(echo "$binaryname --dicomdir $f/DICOMconverted --subject $subject_id --coreg_rigid $param_rigid --coreg_nonrigid $param_BSpline")
    echo $xtermcmd
    ret=$(eval $xtermcmd)
    if [ "$?" -eq "0" ]
    then
        echo "SUCCESS"
    else
        echo "FAILURE"
        continue
    fi
    xtermcmd=$(echo "$binaryname2 --dicomdir $f/DICOMconverted --subject $subject_id")
    echo $xtermcmd
    ret=$(eval $xtermcmd)
    if [ "$?" -eq "0" ]
    then
        echo "SUCCESS"
    else
        echo "FAILURE"
    fi

    let "round = $round + 1"

#    if [ "$round" -gt "3" ]
#    then
#        break
#    fi

done
