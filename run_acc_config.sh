# Extract arguments
start="$1"
stop="$2"
step="$3"
acc_threshold="$4"

python plot_generation.py $start $stop $step $acc_threshold
python unpack.py $start $stop $step $acc_threshold 0 1