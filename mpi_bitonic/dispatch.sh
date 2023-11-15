#!/bin/bash
#Can run all jobs but will run out of Grace credits! 
# declare -a procs=("2" "64" "128" "256" "512" "1024")
#Change time, nodes, tasks/node, and mem for each procs size!!! 
#Single proc size run
declare -a procs=("1024")
# var=("1")
for (( option=1; option <= 4; option++ ));
do
    #echo -e "\n {$option} \n"
    for num_procs in "${procs[@]}";
    do
        # echo $num_procs
        for((expon = 16; expon <= 28; expon+=2));
        # for((i=1; i <=7; i++))
        do
            # echo $((var++))
            # echo "sbatch bitonic.grace_job $num_procs $((2 ** $expon)) $option"
            sbatch bitonic.grace_job $num_procs $((2 ** $expon)) $option
        done
    done
done