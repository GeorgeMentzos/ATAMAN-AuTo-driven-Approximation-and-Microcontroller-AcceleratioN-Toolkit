from tqdm import tqdm
import argparse
import sys 
import importlib.util
import os 

# adding global_lib_path to the system path
global_lib_path = r"C:\Users\mentz\Desktop\TinyML_Approx_and_Unpack\global_libs"
# global_lib_path= "/mnt/c/Users/mentz/Desktop/DriveSync/Thesis/OPTIMIZATIONS/gloabal_libraries"
sys.path.insert(0, global_lib_path)

import myfuncs as mf
import preload_mat_mult as pmm

def create_dir(directory):
    parent_dir = os.getcwd()
    path = os.path.join(parent_dir, directory)
    os.makedirs(path, exist_ok=True)

def get_exclude_arrays(significance_index_file_path):
    spec = importlib.util.spec_from_file_location("significance_indexes", significance_index_file_path)
    significance_indexes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(significance_indexes)

    exclude0 = [
        # significance_indexes.layer0_common_values_below_threshold0,
        significance_indexes.layer1_consecutive_common_values_below_threshold0,
        significance_indexes.layer2_consecutive_common_values_below_threshold0,
        # significance_indexes.layer3_common_values_below_threshold0,
        # significance_indexes.layer4_common_values_below_threshold0
    ]
    exclude1 = [
        # significance_indexes.layer0_common_values_below_threshold1,
        significance_indexes.layer1_consecutive_common_values_below_threshold1,
        significance_indexes.layer2_consecutive_common_values_below_threshold1,        
        # significance_indexes.layer3_common_values_below_threshold1,
        # significance_indexes.layer4_common_values_below_threshold1
    ]

    return exclude0, exclude1

# import the neccesarry info from .tflite file 
model_name = "LeNet_Cifar10_INT8.tflite"
# using the global folder path since lib is in another folder
folder_path = "./tflite_models/"
model_path = folder_path + model_name

# import model parameters
tensor_weights,num_rows,num_cols,bias_values = pmm.extract_weights_np(model_path)


# NOTE REPLACE WITH AUTOMATIC INDEXES BASED ON TFLITE MODEL NOT HARDCODING
# difference between a0 and a1 is that they are one row appart
# .tflite files have flatbuffers at indexes corresponding to the conv layers
# for this example we will manually hardcode the layer indexes of the .tflite file
# as it stands our function uses the location from neutron + 1 as indexs
# indexes for layer 0,1,2
# layer_indexes = [16,14,12,10,8]
layer_indexes = [12,10,8]
factor = 1
layers = 3
skip_layers = 1
layer_indexes = layer_indexes[skip_layers:]
tensor_weights,num_rows,num_cols,bias_values = tensor_weights[skip_layers:],num_rows[skip_layers:],num_cols[skip_layers:],bias_values[skip_layers:]

# # exclude0 contains the common indexes that map to a0 in mat_mult_kernel 
# exclude0 = [significance.layer0_common_values_below_threshold0, significance.layer1_common_values_below_threshold0, significance.layer2_common_values_below_threshold0]
# # exclude1 contains the common indexes that map to a1 in mat_mult_kernel 
# exclude1 = [significance.layer0_common_values_below_threshold1, significance.layer1_common_values_below_threshold1, significance.layer2_common_values_below_threshold1]
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process start and stop arguments.")
    parser.add_argument("start", type=int, help="Start value for the range")
    parser.add_argument("stop", type=int, help="Stop value for the range")
    parser.add_argument("step", type=int, nargs="?", default=1, help="Step value for the range")
    args = parser.parse_args()

    start = args.start
    stop = args.stop
    step = args.step

    total_iterations = ((stop - start)//step) ** (layers-skip_layers)
    progress_bar = tqdm(total=total_iterations, desc="Progress")

    modded_folder_name = "modified_tlfite_models_{}_{}_{}".format(start/1000,(stop-1)/1000,step)
    create_dir(modded_folder_name)
    siginificance_folder_name = "significance_folder_{}_{}_{}".format(start/1000,(stop-1)/1000,step)

for i in [float(ij) / 1000 for ij in range(start, stop, step)]:
    for j in [float(jj) / 1000 for jj in range(start, stop, step)]:
        # for k in [float(kk) / 1000 for kk in range(start, stop, step)]:
        #     for l in [float(ll) / 1000 for ll in range(start, stop, step)]:
                # import the neccesarry info from .tflite file 
                threshold = [-10,i*factor,j]
                significance_index_file_path = "./" + siginificance_folder_name + "/significance_indexes_[{},{},{}].py".format(threshold[0],threshold[1],threshold[2])
                # significance_index_file_path = "./" + siginificance_folder_name + "/significance_indexes_[{},{},{},{},{}].py".format(threshold[0],threshold[1],threshold[2],threshold[3],threshold[4])
                exclude0,exclude1 = get_exclude_arrays(significance_index_file_path)
                # modded_model_name = "modded_" + model_name + "_{},{},{},{},{}".format(threshold[0],threshold[1],threshold[2],threshold[3],threshold[4])
                modded_model_name = "modded_" + model_name + "_{},{},{}".format(threshold[0],threshold[1],threshold[2])
                # using the global folder path since lib is in another folder
                modded_folder_path = "./" + modded_folder_name + "/"
                modded_model_path = modded_folder_path + modded_model_name + ".tflite"
                mf.modify_tflite_weights(layer_indexes, num_cols, exclude0, exclude1, model_path ,modded_model_path )
                progress_bar.update(1)

# Close the progress bar when done
progress_bar.close()