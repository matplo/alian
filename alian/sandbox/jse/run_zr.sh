#!/bin/bash

# run in parallel ./pythia_zr.py for two settings of --py-pthatmin 90 and 100

function run() 
{
		local pthatmin=$1
		local output="pythia_zr_pthatmin${pthatmin}.root"
		echo "Running with pthatmin=${pthatmin}, output=${output}"
		pthatmax=$((pthatmin + 20))
		python3 pythia_zr.py --nev 10000 --py-hardQCD --py-pthatmin ${pthatmin} --jet-pt-min ${pthatmin} --jet-pt-max ${pthatmax} --output ${output} 2>&1 | tee "pythia_zr_pthatmin${pthatmin}.log"
}	

# use GNU parallel to run the function for both pthatmin values
export -f run
parallel --progress run ::: 100 120
