import sys 
import argparse
from tqdm import tqdm

import preload_mat_mult as pmm

layers = 3

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process start and stop arguments.")
    parser.add_argument("start", type=int, help="Start value for the range")
    parser.add_argument("stop", type=int, help="Stop value for the range")
    parser.add_argument("step", type=int, nargs="?", default=1, help="Step value for the range")
    parser.add_argument("acc_threshold", type=float, help="Accuracy threshold value")
    parser.add_argument("shift", type=int, help="Preload with shift", nargs='?', default=0)
    parser.add_argument("dsp", type=int, help="Preload with dsp", nargs='?', default=1)

    args = parser.parse_args()

    start = args.start
    stop = args.stop
    step = args.step
    acc_threshold = args.acc_threshold
    shift = args.shift
    dsp = args.dsp

    model_folder = "./plot_{}_{}_{}".format(start/1000,stop/1000,step)
    model_name = "/optimal_model_{}".format(acc_threshold)
    model_path = model_folder + model_name + ".tflite"

    # if(acc_threshold==0):
    #     model_path = "./tflite_models/LeNet_Cifar10_INT8.tflite"

    tensor_weights,num_rows,num_cols,bias_values = pmm.extract_weights(model_path)
    print(num_rows)

    
    total_iterations = layers
    progress_bar = tqdm(total=total_iterations, desc="Progress")

    for layer in range(layers):
        pmm.preload_layer(layer,num_rows,num_cols,tensor_weights,int(num_rows[layer]/2),num_cols[layer],shift=shift,acc_threshold=acc_threshold,DSP=dsp)
        progress_bar.update(1)

    # Close the progress bar when done
    progress_bar.close()