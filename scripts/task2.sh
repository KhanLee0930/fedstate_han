#!/bin/bash
source $(dirname $(readlink -f $0))/env

mkdir -p $OUTPUT_DIR

# Base Config
N=20
p=10
E=10
B=32
num_rounds=1000
alpha=0.1

# Base Experiment
#for method in Balance FedAvg
#do
#	save_dir=convergence_results/$N-$p-$E-$B-$alpha-$method
#	qsub -l walltime=$walltime -v N=$N,p=$p,E=$E,B=$B,num_rounds=$num_rounds,alpha=$alpha,method=$method,save_dir=$save_dir submit.sh
#done

# alpha ablation
# for method in Balance IID Non-IID Random
for method in Balance IID Non-IID Random
do
	for alpha in 0.1
	do
		save_dir=$OUTPUT_DIR/convergence_results/$N-$p-$E-$B-$alpha-$method
		echo -e "\n\n\n running $method with alpha $alpha"

		python -W ignore::DeprecationWarning ${FEDSATE_SCRIPT} \
			--N $N --p $p --E $E --B $B --num-rounds $num_rounds --alpha $alpha --method $method --save-dir $save_dir 
	done
done

