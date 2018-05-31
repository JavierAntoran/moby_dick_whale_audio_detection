#!/bin/bash

file=$1
n_jobs=$2

#-------------------------------------------------------------------
# create dir
mkdir -p .condor/

#-------------------------------------------------------------------
# base filename

filename_base="${file%.*}"

#-------------------------------------------------------------------
# create condor config

condor_task=.condor/${filename_base}.sub

#::::::::::::::::::::::::::::::::::::::::::
cat > ${condor_task} << EOF

executable = `which python`
arguments  = ${file} \$(icluster)
output     = .condor/${filename_base}_\$(ifile).log
error      = .condor/${filename_base}_\$(ifile).err
log        = .condor/${filename_base}.clog

EOF
#::::::::::::::::::::::::::::::::::::::::::

for((i=1;i<=$n_jobs;i++)); do
	  ifile=`echo $i | awk '{ printf("%03d",$1);}'`
      echo "
      ifile = $ifile
      icluster = $i
      queue " 
done >> ${condor_task}

#-------------------------------------------------------------------
# view the condor config

#cat ${condor_task}

#-------------------------------------------------------------------
# launch condor job

condor_submit.sh --sync y --config ${condor_task} 


#-------------------------------------------------------------------
# clean

#rm -rf .condor


