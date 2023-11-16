#!/bin/bash

for (( option=1; option <= 4; option++ ));
do
    #echo -e "\n {$option} \n"
    for(( num_threads=64; num_threads <= 1024; num_threads<<=1));
    do
        #echo $num_threads
        for((expon = 16; expon <= 28; expon+=2));
        do
            #echo "sbatch bitonic.grace_job $num_threads $((2 ** $expon)) $option"
            sbatch bitonic.grace_job $num_threads $((2 ** $expon)) $option
        done
    done
done