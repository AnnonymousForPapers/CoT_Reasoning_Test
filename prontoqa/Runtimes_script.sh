#!/bin/bash

#SBATCH --ntasks=1    # Job only requires 1 CPU core
#SBATCH --output=R-%x_%j.out

printf "gpt2"
sacct -j 11468096 --format=JobID,Elapsed
sacct -j 11468134 --format=JobID,Elapsed
sacct -j 11468165 --format=JobID,Elapsed
sacct -j 11468187 --format=JobID,Elapsed
sacct -j 11468212 --format=JobID,Elapsed
sacct -j 11468242 --format=JobID,Elapsed

printf "SmolLM2-135M"
sacct -j 11091026 --format=JobID,Elapsed
sacct -j 11091042 --format=JobID,Elapsed
sacct -j 11091050 --format=JobID,Elapsed
sacct -j 11091062 --format=JobID,Elapsed
sacct -j 11091069 --format=JobID,Elapsed
sacct -j 11091074 --format=JobID,Elapsed

printf "SmolLM2-135M-Instruct"
sacct -j 11091091 --format=JobID,Elapsed
sacct -j 11091098 --format=JobID,Elapsed
sacct -j 11091104 --format=JobID,Elapsed
sacct -j 11091614 --format=JobID,Elapsed
sacct -j 11091620 --format=JobID,Elapsed
sacct -j 11091624 --format=JobID,Elapsed

printf "OpenELM-270M"
sacct -j 11084779 --format=JobID,Elapsed
sacct -j 11084825 --format=JobID,Elapsed
sacct -j 11084832 --format=JobID,Elapsed
sacct -j 11084842 --format=JobID,Elapsed
sacct -j 11084850 --format=JobID,Elapsed
sacct -j 11084870 --format=JobID,Elapsed

printf "OpenELM-270M-Instruct"
sacct -j 11085344 --format=JobID,Elapsed
sacct -j 11085365 --format=JobID,Elapsed
sacct -j 11085376 --format=JobID,Elapsed
sacct -j 11085383 --format=JobID,Elapsed
sacct -j 11085396 --format=JobID,Elapsed
sacct -j 11085403 --format=JobID,Elapsed

printf "gpt2-medium"
sacct -j 11476675 --format=JobID,Elapsed
sacct -j 11476695 --format=JobID,Elapsed
sacct -j 11476717 --format=JobID,Elapsed
sacct -j 11476748 --format=JobID,Elapsed
sacct -j 11476779 --format=JobID,Elapsed
sacct -j 11476812 --format=JobID,Elapsed

printf "SmolLM2-360M"
sacct -j 11090917 --format=JobID,Elapsed
sacct -j 11090923 --format=JobID,Elapsed
sacct -j 11090933 --format=JobID,Elapsed
sacct -j 11090939 --format=JobID,Elapsed
sacct -j 11090942 --format=JobID,Elapsed
sacct -j 11090946 --format=JobID,Elapsed

printf "SmolLM2-360M-Instruct"
sacct -j 11090978 --format=JobID,Elapsed
sacct -j 11090989 --format=JobID,Elapsed
sacct -j 11090991 --format=JobID,Elapsed
sacct -j 11090996 --format=JobID,Elapsed
sacct -j 11091004 --format=JobID,Elapsed
sacct -j 11091009 --format=JobID,Elapsed

printf "OpenELM-450M"
sacct -j 11082604 --format=JobID,Elapsed
sacct -j 11082747 --format=JobID,Elapsed
sacct -j 11082788 --format=JobID,Elapsed
sacct -j 11082804 --format=JobID,Elapsed
sacct -j 11082816 --format=JobID,Elapsed
sacct -j 11082828 --format=JobID,Elapsed

printf "OpenELM-450M-Instruct"
sacct -j 11083803 --format=JobID,Elapsed
sacct -j 11083836 --format=JobID,Elapsed
sacct -j 11083852 --format=JobID,Elapsed
sacct -j 11083870 --format=JobID,Elapsed
sacct -j 11083888 --format=JobID,Elapsed
sacct -j 11083899 --format=JobID,Elapsed

printf "gpt2-large"
sacct -j 11477714 --format=JobID,Elapsed
sacct -j 11477751 --format=JobID,Elapsed
sacct -j 11477786 --format=JobID,Elapsed
sacct -j 11477820 --format=JobID,Elapsed
sacct -j 11477848 --format=JobID,Elapsed
sacct -j 11477876 --format=JobID,Elapsed

printf "OpenELM-1_1B"
sacct -j 10615154 --format=JobID,Elapsed
sacct -j 10615174 --format=JobID,Elapsed
sacct -j 10615195 --format=JobID,Elapsed
sacct -j 10615221 --format=JobID,Elapsed
sacct -j 10615246 --format=JobID,Elapsed
sacct -j 10615286 --format=JobID,Elapsed

printf "OpenELM-1_1B-Instruct"
sacct -j 10616780 --format=JobID,Elapsed
sacct -j 10616802 --format=JobID,Elapsed
sacct -j 10616818 --format=JobID,Elapsed
sacct -j 10616860 --format=JobID,Elapsed
sacct -j 10616870 --format=JobID,Elapsed
sacct -j 10616886 --format=JobID,Elapsed

printf "TinyLlama_v1_1"
sacct -j 11092089 --format=JobID,Elapsed
sacct -j 11092100 --format=JobID,Elapsed
sacct -j 11092111 --format=JobID,Elapsed
sacct -j 11092114 --format=JobID,Elapsed
sacct -j 11092126 --format=JobID,Elapsed
sacct -j 11092134 --format=JobID,Elapsed

printf "gpt2-xl"
sacct -j 11477938 --format=JobID,Elapsed
sacct -j 11477968 --format=JobID,Elapsed
sacct -j 11477981 --format=JobID,Elapsed
sacct -j 11478012 --format=JobID,Elapsed
sacct -j 11478041 --format=JobID,Elapsed
sacct -j 11478067 --format=JobID,Elapsed

printf "stablelm-2-1_6b"
sacct -j 11091969 --format=JobID,Elapsed
sacct -j 11091974 --format=JobID,Elapsed
sacct -j 11091982 --format=JobID,Elapsed
sacct -j 11091983 --format=JobID,Elapsed
sacct -j 11091993 --format=JobID,Elapsed
sacct -j 11092004 --format=JobID,Elapsed

printf "stablelm-2-zephyr-1_6b"
sacct -j 11092030 --format=JobID,Elapsed
sacct -j 11092038 --format=JobID,Elapsed
sacct -j 11092048 --format=JobID,Elapsed
sacct -j 11092054 --format=JobID,Elapsed
sacct -j 11092066 --format=JobID,Elapsed
sacct -j 11092075 --format=JobID,Elapsed

printf "SmolLM2-1.7B"
sacct -j 11086596 --format=JobID,Elapsed
sacct -j 11086700 --format=JobID,Elapsed
sacct -j 11086733 --format=JobID,Elapsed
sacct -j 11086765 --format=JobID,Elapsed
sacct -j 11086849 --format=JobID,Elapsed
sacct -j 11086884 --format=JobID,Elapsed

printf "SmolLM2-1.7B-Instruct"
sacct -j 11089096 --format=JobID,Elapsed
sacct -j 11089120 --format=JobID,Elapsed
sacct -j 11089142 --format=JobID,Elapsed
sacct -j 11089143 --format=JobID,Elapsed
sacct -j 11089144 --format=JobID,Elapsed
sacct -j 11089168 --format=JobID,Elapsed

printf "gemma_2_2b_it"
sacct -j 10539177 --format=JobID,Elapsed
sacct -j 10539191 --format=JobID,Elapsed
sacct -j 10539202 --format=JobID,Elapsed
sacct -j 10539210 --format=JobID,Elapsed
sacct -j 10539225 --format=JobID,Elapsed
sacct -j 10539238 --format=JobID,Elapsed

printf "OpenELM-3B"
sacct -j 10611674 --format=JobID,Elapsed
sacct -j 10611705 --format=JobID,Elapsed
sacct -j 10611723 --format=JobID,Elapsed
sacct -j 10611739 --format=JobID,Elapsed
sacct -j 10611769 --format=JobID,Elapsed
sacct -j 10611785 --format=JobID,Elapsed

printf "OpenELM-3B-Instruct"
sacct -j 10613684 --format=JobID,Elapsed
sacct -j 10613705 --format=JobID,Elapsed
sacct -j 10613721 --format=JobID,Elapsed
sacct -j 10613739 --format=JobID,Elapsed
sacct -j 10613758 --format=JobID,Elapsed
sacct -j 10613789 --format=JobID,Elapsed

printf "gemma_2_9b_it"
sacct -j 10541073 --format=JobID,Elapsed
sacct -j 10541080 --format=JobID,Elapsed
sacct -j 10541088 --format=JobID,Elapsed
sacct -j 10541095 --format=JobID,Elapsed
sacct -j 10541098 --format=JobID,Elapsed
sacct -j 10541107 --format=JobID,Elapsed
