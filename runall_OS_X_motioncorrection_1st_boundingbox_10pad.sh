#!/bin/sh

#  Script.sh
#
#
#  Created by Harri Merisaari on 3/10/14.
#
set +e
binaryname="python motioncorrection_1st_boundingbox_10pad.py"
selectionfile="namelist_for_noncorrected_pmaps_failed.txt"
DWIbasedir="/Users/eija/Desktop/prostate_MR/PET_MR_dwis/"
logfile=$(echo "failures.txt")
no_folders=(`wc -l $selectionfile`)
no_folder=${no_folders[0]}
echo $no_folders " folders to be processed"
echo "Failed executions" > $logfile
for (( round=1; round<=$no_folders; round++ ))
do
    name1=$(sed -n "$round"p $selectionfile | awk '{print $1}')
    name2=$(sed -n "$round"p $selectionfile | awk '{print $2}')
    f=$(echo $DWIbasedir$name1)
    subject_id=$name2

    xtermcmd=$(echo "$binaryname --dicomdir $f/DICOMconverted --subject $subject_id")
    echo $xtermcmd
    ret=$(eval $xtermcmd)
    if [ "$?" -eq "0" ]
    then
        echo "SUCCESS"
    else
        echo "FAILURE"
        echo "Failure in " + $xtermcmd >> $logfile
    fi
done
