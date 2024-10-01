import sys 
import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
import importlib.util
import argparse
import os

# adding global_lib_path to the system path
# global_lib_path = r"C:\Users\mentz\Desktop\DriveSync\Thesis\OPTIMIZATIONS\gloabal_libraries"
# global_data_path = r"C:\Users\mentz\Desktop\DriveSync\Thesis\OPTIMIZATIONS\global_data"
# global_lib_path = r"C:\Users\mentz\Desktop\TinyML_Approx_and_Unpack\global_libs"
global_data_path = "data_preprocessing"
# global_lib_path= "/mnt/c/Users/mentz/Desktop/DriveSync/Thesis/OPTIMIZATIONS/gloabal_libraries"
# global_data_path= "/mnt/c/Users/mentz/Desktop/DriveSync/Thesis/OPTIMIZATIONS/global_data"
# sys.path.insert(0, global_lib_path)

import myfuncs as mf

def create_dir(directory):
    parent_dir = os.getcwd()
    path = os.path.join(parent_dir, directory)
    os.makedirs(path, exist_ok=True)

def get_mac_reduction_arrays(significance_index_file_path):

    spec = importlib.util.spec_from_file_location("significance_indexes", significance_index_file_path)
    significance_indexes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(significance_indexes)

    total_macs = [
        # significance_indexes.layer_0_total_macs,
        significance_indexes.layer_1_total_macs,
        significance_indexes.layer_2_total_macs,
        # significance_indexes.layer_3_total_macs,
        # significance_indexes.layer_4_total_macs
    ]

    perforated_macs = [
        # significance_indexes.layer_0_perforated_macs,
        significance_indexes.layer_1_perforated_macs_dsp,
        significance_indexes.layer_2_perforated_macs_dsp,
        # significance_indexes.layer_3_perforated_macs,
        # significance_indexes.layer_4_perforated_macs
    ]

    perforation_perchentage = [
          
        significance_indexes.layer_1_perforation_perchentage_dsp,  
        significance_indexes.layer_2_perforation_perchentage_dsp
    ]

    return total_macs, perforated_macs, perforation_perchentage

factor = 1
layers=3
skip_layers=1
#  script to test accuracy of modded .tflite files
inference_data_samples = 1000

model_name = "LeNet_Cifar10_INT8.tflite"
# using the global folder path since lib is in another folder
folder_path = "../models/"
model_path = folder_path + model_name

# modded_model_name = "modded_" + model_name + "_{},{},{},{},{}"
modded_model_name = "modded_" + model_name + "_{},{},{}"
# using the global folder path since lib is in another folder

# orig_acc = mf.tflite_inference(global_data_path,model_path,inference_data_samples)
# np.save("orig_acc.npy",[orig_acc])

# Lists to store the x and y values for the diagram
x_values = []
y_values = []

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
    modded_folder_path = "./" + modded_folder_name + "/"
    modded_model_path = modded_folder_path + modded_model_name + ".tflite"
    siginificance_folder_name = "significance_folder_{}_{}_{}".format(start/1000,(stop-1)/1000,step)

    for i in [float(ij) / 1000 for ij in range(start, stop, step)]:
        for j in [float(jj) / 1000 for jj in range(start, stop, step)]:
            # for k in [float(kk) / 1000 for kk in range(start, stop, step)]:
            #     for l in [float(ll) / 1000 for ll in range(start, stop, step)]:
                    # import the neccesarry info from .tflite file 
                    threshold = [-10, i*factor,j]
                    significance_index_file_path = "./" + siginificance_folder_name + "/significance_indexes_[{},{},{}].py".format(threshold[0],threshold[1],threshold[2])
                    # significance_index_file_path = "./" + siginificance_folder_name + "/significance_indexes_[{},{},{},{},{}].py".format(threshold[0],threshold[1],threshold[2],threshold[3],threshold[4])
                    total_macs,perforated_macs,perforation_perchentage = get_mac_reduction_arrays(significance_index_file_path)
                    acc = mf.tflite_inference(global_data_path,modded_model_path.format(threshold[0],threshold[1],threshold[2]),inference_data_samples,inference_data_samples,dataset="cifar10")
                    # acc = mf.tflite_inference(global_data_path,modded_model_path.format(threshold[0],threshold[1],threshold[2],threshold[3],threshold[4]),inference_data_samples,inference_data_samples,dataset="cifar10")
                    # print(total_macs,perforated_macs)
                    # Calculate the ratio of total perforated MACs to total MACs
                    if(perforation_perchentage[0]<15):
                        perforated_macs[0]=0
                
                    if(perforation_perchentage[1]<15):
                        perforated_macs[1]=0
                    
                    total_perforated_macs = sum(perforated_macs)
                    total_macs_sum = sum(total_macs)
                    mac_ratio = total_perforated_macs / total_macs_sum
                    # Append the x and y values to the lists
                    x_values.append(mac_ratio)
                    y_values.append(acc)

                    progress_bar.update(1)

    # Close the progress bar when done
    progress_bar.close()

    dse_res_folder = "dse_res_{}_{}_{}".format(start/1000,stop/1000,step)
    create_dir(dse_res_folder)
    dse_res_folder_path = "./" + dse_res_folder + "/"

    np.save(dse_res_folder_path + "x_vals_{}_{}_{}.npy".format(start,stop,step),x_values)
    np.save(dse_res_folder_path + "y_vals_{}_{}_{}.npy".format(start,stop,step),y_values)