#!/bin/bash

# run in parallel ./pythia_zr.py for pthatmin 100 and 120, 8 seeds each, max 8 jobs at once

start_seed=12345

function run()
{
		local pthatmin=$1
		local user_seed=$2
		local seed=$((start_seed + user_seed))
		local output="pythia_zr_pthatmin${pthatmin}_seed${user_seed}.root"
		echo "Running with pthatmin=${pthatmin}, seed=${seed}, output=${output}"
		# pthatmax=$((pthatmin + 20))
		jet_pt_max=$((pthatmin + 50))
		# yaspenv 
		python3 pythia_zr.py --nev 10000 --py-hardQCD --py-pthatmin ${pthatmin} --jet-pt-min ${pthatmin} --jet-pt-max ${jet_pt_max} --output ${output} --py-seed ${seed} --py-ecm 5360 2>&1 | tee "pythia_zr_pthatmin${pthatmin}_seed${user_seed}.log"
}

# use GNU parallel to run the function for both pthatmin values
export -f run
export start_seed
# parallel -j8 --progress run ::: 100 118 120 ::: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
parallel -j8 --progress run ::: 115 ::: 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15

hadd -f pythia_zr_pthatmin100.root pythia_zr_pthatmin100_seed*.root
hadd -f pythia_zr_pthatmin120.root pythia_zr_pthatmin120_seed*.root
hadd -f pythia_zr_pthatmin115.root pythia_zr_pthatmin115_seed*.root