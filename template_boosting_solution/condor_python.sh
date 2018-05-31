#!/bin/bash

file=$1

#-------------------------------------------------------------------
# create dir
mkdir -p .condor/

#-------------------------------------------------------------------
# base filename

file_condor_base=`basename $file .py`


#-------------------------------------------------------------------
# create condor config

condor_task=.condor/${file_condor_base}.sub


#-------------------------------------------------------------------
if [ -f condor.cfg ];
then
    echo "condor.cfg exists."
    cp condor.cfg ${condor_task} 
else
   echo "condor.cfg does not exist, creating default"
#::::::::::::::::::::::::::::::::::::::::::
cat > ${condor_task}  << EOF

universe              = vanilla
notification          = Never 
nice_user             = True
getenv                = True
request_cpus          = 1
request_gpus          = 1
request_memory        = 2000 
should_transfer_files = no

EOF
#::::::::::::::::::::::::::::::::::::::::::
fi

sed -i 's:request_gpus.*::g' ${condor_task} 

#-------------------------------------------------------------------

#::::::::::::::::::::::::::::::::::::::::::
cat >> ${condor_task} << EOF

executable = `which python`
arguments  = ${@}
output     = ${file_condor_base}.log
error      = ${file_condor_base}.err
log        = ${file_condor_base}.clog
queue

EOF
#::::::::::::::::::::::::::::::::::::::::::

#-------------------------------------------------------------------
# view the condor config

#cat ${condor_task}

#-------------------------------------------------------------------
# launch condor job

condor_submit.sh --sync y --config ${condor_task} 

#-------------------------------------------------------------------
# clean

#rm -rf .condor/

