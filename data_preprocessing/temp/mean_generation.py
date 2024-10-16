import sys 
# adding global_lib_path to the system path
global_lib_path = r"C:\Users\mentz\Desktop\TinyML_Approx_and_Unpack\global_libs"
sys.path.insert(0, global_lib_path)

import preload_mat_mult as pmm
import optimized_mac_perforation as omp

model_name = "LeNet_Cifar10_INT8"
model_path = r"C:\Users\mentz\Desktop\TinyML_Approx_and_Unpack\models/" + model_name + ".tflite"

tensor_weights,num_rows,num_cols,bias_values = pmm.extract_weights(model_path)

data_len = 1000
layer_num = 3

# NOTE FOR NOW FIRST 10 SAMPLES ARE SKIPPED SO 990 TOTAL SAMPLES FOR AVERAGE APPROX 1000 FOR OUR PUPROSES
# ALSO CHANGE MAT MULT COUNT IN FUNCS
for layer_index in range(1,layer_num):
    omp.mean_distance_metric_np(layer_index,data_len,tensor_weights,num_rows,num_cols,bias_values)