#!/bin/bash
LOG="DISPATCH.log"
NAME="merge"
for (( option=1; option <= 4; option++ ));
do
    #echo -e "\n {$option} \n"
    for(( num_threads=64; num_threads <= 1024; num_threads<<=1));
    do
        #echo $num_threads
        for((expon = 16; expon <= 28; expon+=2));
        do
            echo "sbatch $NAME.grace_job $num_threads $((1<<$expon)) $option"
            sbatch $NAME.grace_job $num_threads $((1<<$expon)) $option  >> $LOG
        done
    done
done