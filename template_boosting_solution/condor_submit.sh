#!/bin/bash

################################################################################
# This prints the help line
function PrintHelp() {
  echo "USAGE: condor_submit.sh [options] [--config FILE || --exec PROG ARGS]";
  echo ""
  echo "  --config FILE                  Condor submission file.";
  echo "  --exec PROG ARGS               Runs executable PROG with arguments ARGS"
  echo ""
  echo "  Options:";
  echo "  -p                             Priority"
  echo "  -r                             RAM"
  echo "  -c                             CPUs"
  echo "  --nice true/false              Sets the nice feature"
  echo "  --short true/false             Sets the ShortJob feature"
  echo "  --big true/false               Sets the BigJob feature"
  echo "  --sync y                       Waits for the jobs to finish"
  echo "  --concurrency_limits label1:n1[,label2:n2,...]   Limits the number of concurrent jobs with the label1 to 1000/n1"
  echo "  --check_np n (10)              Displays progress info each time that n processes finish"
  echo "  --delta_r (500)                Increases the request_memory of hold processes by delta_r and releases them"
  echo "  --output file                  Output file"
  echo "  --error file                   Error file"
  echo "  --log file                     Condor log file"
  echo "  --help                         Prints this help";
#  echo "  --helphelp                     Prints this help + config help";
} > /dev/stderr
################################################################################

################################################################################
# This prints the help line
function PrintHelpHelp() {
  PrintHelp
  echo "";
} > /dev/stderr
################################################################################


################################################################################
# Gets the Number of process running 
function GetNProcs() {
    		
	ERR=1
    while [ $ERR -eq 1 ]
      do
      N_PROCESS=$(condor_q -nobatch $TASK_ID 2> $TASK_FILE_ERR | awk '$2=="'$USER'" || $2=="nice-user.'$USER'" && $6!="C" && $6!="X"' | wc -l)
      ERR=$(awk '/Failed/ || /Error/' $TASK_FILE_ERR | wc -l)
    done
#color    echo -e "\e[1m\e[92m$N_PROCESS\e[21m processes remaining at $(date)\e[0m"
	  echo $N_PROCESS | awk '{ printf("     :\x1b[1;31m%4d\x1b[0m ",$1);}'
	  echo "processes remaining at "$(date)
	  
}
################################################################################

################################################################################
# Gets the condor logs for the condor_wait
function GetLogFiles () {
    ERR=1
    while [ $ERR -eq 1 ]
      do
      R=$(condor_q -nobatch -l $TASK_ID 2> $TASK_FILE_ERR | grep UserLog | sort -u | awk '{ gsub(/"/,"",$3); print $3}')
      read -a LOGS <<<$R
      ERR=$(awk '/Failed/ || /Error/' $TASK_FILE_ERR | wc -l)
    done
    N_LOGS=${#LOGS[@]}
    if [ $N_LOGS -gt 1 ]; then
	echo -e "\e[1m\e[93m
###############  Warning  ######################
Your are using $N_LOGS condor-log files.
Using only one condor log file is more efficient 
for this version of condor_submit.sh
################################################
\e[0m"
    fi

}
################################################################################

################################################################################
# Increases the request_memory of processes on hold
function AddMemToHoldProcs () {
    INFO=$(condor_q -nobatch -hold $TASK_ID | awk '/Job has gone over memory limit/ { printf "%s %s ",$1,$(NF-1)}')
    if [ -z "$INFO" ];then
	return
    fi
    IDS=""
    ID=""
    for f in $INFO
      do
      if [ -z "$ID" ];then
	  ID=$f
      else
	  MEM=$(echo $f | awk '{ print $1+'$DMEM'}')      
	  condor_qedit $ID RequestMemory $MEM
	  IDS="$IDS $ID"
	  ID=""
      fi
    done
    if [ ! -z "$IDS" ];then
	condor_release $IDS
    fi
}
################################################################################

################################################################################
# Waits for all the processes to finish
function WaitProcs () {

    GetNProcs
    GetLogFiles

    if [ $N_PROCESS -lt $CHECK_NP ];then
	CHECK_NP=$N_PROCESS
    fi
    CHECK=$CHECK_NP

    read -a TASK_ID_V <<<$TASK_ID
    N_TASK_ID=${#TASK_ID_V[@]}

    if [ $N_LOGS -eq 1 ];then
	while [ $N_PROCESS -ne 0 ]
	  do
	  if [ $N_TASK_ID -eq 1 ];then 
	      condor_wait -num $CHECK ${LOGS[0]} $TASK_ID >  /dev/null
	  else
	      condor_wait -num $CHECK ${LOGS[0]} 		  >  /dev/null
	  fi
	  GetNProcs
	  if [ $N_PROCESS -lt $CHECK_NP ];then
	      CHECK_NP=$N_PROCESS
	  fi
	  let "CHECK=CHECK+CHECK_NP"
	  AddMemToHoldProcs
	done
    else
	for((i=0;i<$N_LOGS;i++))
	do
	  if [ $N_TASK_ID -eq 1 ];then 
	      condor_wait -status ${LOGS[$i]} $TASK_ID
	  else
	      condor_wait -status ${LOGS[$i]} 
	  fi
	  GetNProcs
	  if [ $N_PROCESS -eq 0 ];then
	      break
	  fi
	  AddMemToHoldProcs
	done
    fi

}

# condor_wait -num $CHECK_NP ${LOGS[0]} | \
# 		  awk -v ids="$TASK_ID" '
#                       BEGIN{ nids=split(ids,ids_v," ")} 
#                            { for(i=1;i<=nids;i++){ 
#                                  if($1 ~ ids_v[i]){ print $0; break}
#                              }
#                            }'
	 

################################################################################

LOCAL_CONF_FILE=$PWD/condor.cfg
GLOBAL_CONF_FILE=$HOME/.condor/condor.cfg

#---------------------------------------------------------------
# in case no parameters are specified, exit with help screen
if [ $# -eq 0 ]; then
  echo "No arguments specified"
  PrintHelp
  exit
elif [ $# -eq 1 ]; then
  if [ "$1" = "--help" ]; then
    PrintHelp
    exit
  elif [ "$1" = "--helphelp" ]; then
    PrintHelpHelp
    exit
  fi
fi


#---------------------------------------------------------------
SYNC=false
SUB=false
CHECK_NP=10
DMEM=500
CLIMITS=""
# parse argument list

while [ $# -ge 2 ]; do
  case $1 in
      # read options
      -p)           PRIO=$2;     shift;;
      -r)           RAM=$2;      shift;;
      -c)           NCPU=$2;     shift;;
      --nice)       NICE=$2;     shift;;
      --short)      SHORT=$2;    shift;;
      --big)        BIG=$2;      shift;;
      --check_np)   CHECK_NP=$2; shift;;
      --delta_r)    DMEM=$2;     shift;;
      --sync) 
	  if [[ $2 == "y" || $2 == "true"  ]];then
	      SYNC=true;
	  fi
	  shift;;
      --concurrency_limits)
	  CLIMITS=$2;
	  shift;;
      --output)     OUTPUT=$2;    shift;;
      --error)      ERROR=$2;     shift;;
      --log)        LOG=$2;       shift;;
      # read extra config file
      --sub)        TASK_FILE=$2; SUB=true; shift;;
      --config)     CONF_FILE=$2; shift;;
      --exec)       PROG=$2;      shift; shift
	  ARGS="";
	  while [ $# -ge 1 ]; do
	      ARGS="$ARGS $1"
	      shift;
	  done
	  ;;
      *) 
	  echo "Unknown option $1"
	  PrintHelp
	  exit
  esac
  shift
done

#---------------------------------------------------------------
if [ $SUB == "true" ];then
	TASK_FILE_ERR=${TASK_FILE%.*}.err
	
else
	TMPDIR=.condor
	if [ ! -d $TMPDIR ];then
    	mkdir -p $TMPDIR
	fi
	TASK_FILE=$TMPDIR/condor_task.$$

	TASK_FILE_ERR=$TASK_FILE.err

 	# load config files
	if [ -e "$LOCAL_CONF_FILE" ]; then
    	cat $LOCAL_CONF_FILE
	else
    	if [ -e "$GLOBAL_CONF_FILE" ]; then
		cat $GLOBAL_CONF_FILE
    	fi
	fi > $TASK_FILE 


	if [ -n "$PRIO" ];then
    	echo "priority = $PRIO" >> $TASK_FILE
	fi


	if [ -n "$NCPU" ];then
    	echo "request_cpus = $NCPU" >> $TASK_FILE
	fi

	if [ -n "$RAM" ];then
    	echo "request_memory = $RAM" >> $TASK_FILE
	fi

	if [ -n "$SHORT" ];then
    	echo "+ShortJob = $SHORT" >> $TASK_FILE
	fi

	if [ -n "$BIG" ];then
    	echo "+BigJob = $BIG" >> $TASK_FILE
	fi

	if [ -n "$NICE" ];then
    	echo "nice_user = $NICE" >> $TASK_FILE
	fi


	if [ -n "$CLIMITS" ];then
	  echo "concurrency_limits = $CLIMITS" >> $TASK_FILE
	fi

	if [ -n "$OUTPUT" ];then
    	echo "output = $OUTPUT" >> $TASK_FILE
	fi

	if [ -n "$ERROR" ];then
    	echo "error = $ERROR" >> $TASK_FILE
	fi

	if [ -n "$LOG" ];then
    	echo "log = $LOG" >> $TASK_FILE
	fi

	if [ -e "$CONF_FILE" ];then
    	cat $CONF_FILE >> $TASK_FILE
	fi

	if [ -n "$PROG" ];then
    	BPROG=$(basename $PROG)
	#    NAME=$(echo "$BPROG $ARGS" | sed -e 's@^\s*@@' -e 's@\s*$@@' -e 's@\s\+@.@g' -e 's@/@.@g')
    	NAME=$BPROG
    	echo "executable = $PROG" 
    	echo "arguments = \"$ARGS\"" 
    	if [ ! -n "$OUTPUT" ];then
		echo "output = $TASK_FILE.$NAME.log"
    	fi
    	if [ ! -n "$ERROR" ];then
		echo "error = $TASK_FILE.err" 
    	fi
    	if [ ! -n "$LOG" ];then
		echo "log = $TASK_FILE.clog" 
    	fi
    	echo "queue" 
	fi >> $TASK_FILE

fi

echo "Submitting $TASK_FILE"
#cat $TASK_FILE
TASK_ID=$(condor_submit $TASK_FILE | awk '/cluster / && !/HERMES/ { sub(/.*cluster /,""); sub(/\.*$/,"");  printf "%s ",$0}')
#color echo "Submitted clusters $TASK_ID"
echo " + submitted cluster"
echo $TASK_ID   | awk '{ printf("   - condor_id:  \x1b[1;32m%d\x1b[0m\n",$1);}'
NCLUSTERS=`grep 'queue' $TASK_FILE | wc | awk '{ print $1 }'`
echo $NCLUSTERS | awk '{ printf("   - n_jobs:     \x1b[32m%d\x1b[0m\n",$1);}'
echo $PRIO      | awk '{ printf("   - priority:   \x1b[32m%d\x1b[0m\n",$1);}'
echo $NCLUSTERS | awk '{ printf("\x1b[31m   + condor: x%d processes\x1b[0m\n",$1);}'


if [ -n "$TASK_ID" ];then
#     if [ -n "$PRIO" ];then
# 	condor_prio -p $PRIO $TASK_ID
#     fi
    if [ $SYNC == "true" ];then
	WaitProcs
    fi
fi

if [ $SUB == "false" ];then
	rm -f $TASK_FILE
fi


echo -n "   - n_errors: "

BASENAME=${TASK_FILE%.*}
N_ERR=`ls -l $BASENAME*err | awk '{if($5 != 0) {print $5}}' | wc | awk '{ print $1}'`
if [ $N_ERR == 0 ] 
then
	echo $N_ERR | awk '{ printf("  \x1b[1;32m%d\x1b[0m\n",$1);}'
else
	echo $N_ERR | awk '{ printf("  \x1b[1;31m%d\x1b[0m\n",$1);}'
	exit -1
fi

