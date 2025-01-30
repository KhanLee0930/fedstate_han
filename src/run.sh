#! /bin/sh

walltime=72:00:00

# Base Config
N=10
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
for method in Balance FedAvg
do
	for alpha in 1 10 100
	do
		save_dir=convergence_results/$N-$p-$E-$B-$alpha-$method
		qsub -l walltime=$walltime -v N=$N,p=$p,E=$E,B=$B,num_rounds=$num_rounds,alpha=$alpha,method=$method,save_dir=$save_dir submit.sh
	done
done

