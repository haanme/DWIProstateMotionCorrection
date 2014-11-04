#!/bin/sh

#  Script.sh
#  
#
#  Created by Harri Merisaari on 3/10/14.
#
set +e
binaryname="python motioncorrection_2nd.py"
binaryname2="python motioncorrection_transformix.py"
selectionfile="namelist.txt"
DWIbasedir="/Users/eija/Desktop/prostate_MR/PET_MR_dwis/"

param_rigid=$(echo "Par0001translation_mod_ver"$1".txt")
param_BSpline=$(echo "Par0001bspline08_mod_ver"$1".txt")
output_dir=$(echo "/Users/eija/Desktop/prostate_MR/motioncorrected_ver"$1)
logfile=$(echo "failures_motioncorrected_ver"$1".txt")

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

    xtermcmd=$(echo "$binaryname --dicomdir $f/DICOMconverted --subject $subject_id --coreg_rigid $param_rigid --coreg_nonrigid $param_BSpline")
    echo $xtermcmd
    ret=$(eval $xtermcmd)
    if [ "$?" -eq "0" ]
    then
        echo "SUCCESS"
    else
        echo "FAILURE"
        echo "Failure in " + $xtermcmd >> $logfile
        continue
    fi
    xtermcmd=$(echo "$binaryname2 --dicomdir $f/DICOMconverted --subject $subject_id --outdir $output_dir")
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
cp $param_rigid $output_dir
cp $param_BSpline $output_dir
cp $logfile $output_dir
cp $0 $output_dir
