#!/bin/bash
source $(dirname $(readlink -f $0))/env

mkdir -p $OUTPUT_DIR

# Base Experiment
#for method in Balance FedAvg
#do
#	save_dir=convergence_results/$N-$p-$E-$B-$alpha-$method
#	qsub -l walltime=$walltime -v N=$N,p=$p,E=$E,B=$B,num_rounds=$num_rounds,alpha=$alpha,method=$method,save_dir=$save_dir submit.sh
#done

# alpha ablation
# for method in Balance IID Non-IID Random

#for method in Random
#do
#	for alpha in 0.2
#	do
#		save_dir=$OUTPUT_DIR/convergence_results/fed_prox$N-$p-$E-$B-$alpha-$method
#		mkdir -p $save_dir
#		echo -e "\n\n\n running $method with alpha $alpha"
#		python -W ignore::DeprecationWarning ${FEDSATE_SCRIPT} \
#			--N $N --p $p --E $E --B $B --num-rounds $num_rounds --alpha $alpha --method $method --save-dir $save_dir
#	done
#done
N=20
p=10
E=10
B=32
num_rounds=1000
for method in Random
do
	for alpha in 0.2
	do
	  PROJECT_ROOT=/home/svu/e1143336/fedstate
    OUTPUT_DIR=${PROJECT_ROOT}/output
		save_dir=$OUTPUT_DIR/convergence_results/hybrid2-$N-$p-$E-$B-$alpha-$method
		FEDSATE_SCRIPT=/home/svu/e1143336/fedstate/src/main_hybrid_training.py
		mkdir -p $save_dir
		echo -e "\n\n\n running $method with alpha $alpha"
		python -W ignore::DeprecationWarning ${FEDSATE_SCRIPT} \
			--N $N --p $p --E $E --B $B --num-rounds $num_rounds --alpha $alpha --method $method --save-dir $save_dir
	done
done
