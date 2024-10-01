from tqdm import tqdm
import argparse
import os
import sys 

# adding global_lib_path to the system path
# global_lib_path = r"C:\Users\mentz\Desktop\DriveSync\Thesis\OPTIMIZATIONS\gloabal_libraries"
# global_data_path = r"C:\Users\mentz\Desktop\DriveSync\Thesis\OPTIMIZATIONS\global_data\cifar10_5_layer_datageneration"
# global_lib_path= "/mnt/c/Users/mentz/Desktop/DriveSync/Thesis/OPTIMIZATIONS/gloabal_libraries"
# global_data_path="/mnt/c/Users/mentz/Desktop/DriveSync/Thesis/OPTIMIZATIONS/global_data/cifar10_5_layer_datageneration"
#global_lib_path = "~/LeNet/global_libs"
#global_data_path = "~/LeNet/data_preprocessing"
#sys.path.insert(0, global_lib_path)

global_data_path = "./data_preprocessing"

import optimized_mac_perforation as omp
import preload_mat_mult as pmm

def create_dir(directory):
    parent_dir = os.getcwd()
    path = os.path.join(parent_dir, directory)
    os.makedirs(path, exist_ok=True)

# import the neccesarry info from .tflite file 
model_name = "LeNet_Cifar10_INT8"
# using the global folder path since lib is in another folder
folder_path = "./tflite_models/"
model_path = folder_path + model_name + ".tflite"

# import model parameters
tensor_weights,num_rows,num_cols,bias_values = pmm.extract_weights_np(model_path)

# num of layers
layer = 3
# total number of samples used to calculate the mean
data_len = 1000
factor = 1
skip_layers = 1

# loop for getting all threshold vals within a certain range
# for this example we will iterate over all layer1 and layer2
# all threshold vals will range between 0 and 0.01 with a step of 0.001 
# we will test all combinations for layer1 and layer2 we need 2 for loops
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process start and stop arguments.")
    parser.add_argument("start", type=int, help="Start value for the range")
    parser.add_argument("stop", type=int, help="Stop value for the range")
    parser.add_argument("step", type=int, nargs="?", default=1, help="Step value for the range")
    args = parser.parse_args()

    start = args.start
    stop = args.stop
    step = args.step

    total_iterations = ((stop - start)//step) ** (layer-skip_layers)
    progress_bar = tqdm(total=total_iterations, desc="Progress")

    siginificance_folder_name = "significance_folder_{}_{}_{}".format(start/1000,(stop-1)/1000,step)
    create_dir(siginificance_folder_name)

    for i in [float(ij) / 1000 for ij in range(start, stop, step)]:
        for j in [float(jj) / 1000 for jj in range(start, stop, step)]:            
            # for k in [float(kk) / 1000 for kk in range(start, stop, step)]:
            #     for l in [float(ll) / 1000 for ll in range(start, stop, step)]:
                    # import the neccesarry info from .tflite file 
                    threshold = [-10, i*factor,j]
                    # threshold = [-10, i,j,k,l]
                    # threshold = [-10,i*factor,j]
                    # for now common thresholds
                    # we choose -10 for layer 0 to gurantee that nothing will be excluded
                    # threshold = [-10,i,j]
                    significance_path = "./" + siginificance_folder_name + "/significance_indexes_[{},{},{}].py".format(threshold[0],threshold[1],threshold[2])
                    # significance_path = "./" + siginificance_folder_name + "/significance_indexes_[{},{},{},{},{}].py".format(threshold[0],threshold[1],threshold[2],threshold[3],threshold[4])
                    # calculate the significance based on the precomputed mean values as well as thresholds for exclude arrays
                    omp.significance(layer, data_len, threshold, threshold, num_rows, num_cols, global_data_path, significance_path,skip_layer=skip_layers)
                    progress_bar.update(1)

    # Close the progress bar when done
    progress_bar.close()

