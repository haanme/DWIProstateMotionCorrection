#!/bin/sh

#  Script.sh
#  
#
#  Created by Harri Merisaari on 3/10/14.
#
set +e
binaryname="python motioncorrection_transformix.py"
no_folders=$(ls -l -d ../PET_MR_dwis/* | grep '^d' | grep -v 'Carimas27' | grep -v 'ProjectTemplate' | awk '{print $9}' | wc -l)
folders=$(ls -l -d ../PET_MR_dwis/* | grep '^d' | grep -v 'Carimas27' | grep -v 'ProjectTemplate' | awk '{print $9}')
# start
echo $no_folders " folders to be processed"
let "round = 1"
for f in $folders
do
    subject_id=$(echo $f | awk -F '/' '{print $3}')

    subject_id=$(echo "10_moring_hB_1a")
    f=$(echo "../PET_MR_dwis/10_moring_hB_1a")

    xtermcmd=$(echo "$binaryname --dicomdir $f/DICOMconverted --subject $subject_id")

    echo $xtermcmd

    ret=$(eval $xtermcmd)
    if [ "$?" -eq "0" ]
    then
        echo "SUCCESS"
    else
        echo "FAILURE"
    fi
    let "round = $round + 1"

    break

done
