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