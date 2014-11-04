#!/bin/sh

#  Script.sh
#
#
#  Created by Harri Merisaari on 3/10/14.
#
set +e
binaryname="python motioncorrection_3rd_DICOM2ASCII10.py"
selectionfile="namelist_for_noncorrected_pmaps.txt"
basedir="/Users/eija/Desktop/prostate_MR/pipelinedata"
output_dir="/Users/eija/Desktop/prostate_MR/ASCII_noncorrected_for_pmaps"
logfile="failures.txt"

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

    xtermcmd=$(echo "$binaryname --basedir $basedir --outdir $output_dir --subject $subject_id --subdir Noncorrected10")
    echo $xtermcmd
    ret=$(eval $xtermcmd)
    if [ "$?" -eq "0" ]
    then
        echo "SUCCESS"
    else
        echo "FAILURE"
        echo "Failure in " + $xtermcmd >> $logfile
        break
    fi
done
