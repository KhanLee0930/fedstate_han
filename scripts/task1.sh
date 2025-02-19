N=30
p=10
E=10
B=32
seed=0
num_rounds=2000
method=Non-IID
alpha=0.1
PROJECT_ROOT=/home/svu/e1143336/fedstate
OUTPUT_DIR=${PROJECT_ROOT}/output
save_dir=$OUTPUT_DIR/convergence_results
FEDSATE_SCRIPT=/home/svu/e1143336/fedstate/src/main_data_sharing.py

for seed in 0 10 20
do
  echo -e "\n\n\n running $method with alpha $alpha with seed $seed"
  python -W ignore::DeprecationWarning ${FEDSATE_SCRIPT} \
        --N $N \
        --p $p \
        --E $E \
        --B $B \
        --num-rounds $num_rounds \
        --alpha $alpha \
        --method $method \
        --save-dir $save_dir \
        --seed $seed
done