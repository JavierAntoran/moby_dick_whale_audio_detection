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
output     = ${filename_base}_\$(ifile).log
error      = ${filename_base}_\$(ifile).err
log        = ${filename_base}.clog
universe              = vanilla
notification          = Never 
nice_user             = True
getenv                = True
request_cpus          = 1
request_gpus          = 0
request_memory        = 1500 
should_transfer_files = no

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


