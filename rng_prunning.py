import tensorflow as tf

# Load the TFLite model
tflite_model_path = './tflite_models/LeNet_Cifar10_INT8.tflite'

import myfuncs as mf
import preload_mat_mult as pmm

# zero_perch = 70
zero_perch = [5,10,15,20,30,50]
num_layers = 3
layer_indexes = [12,10,8]
# layer_indexes = [16,14,12,10,8]
tensor_weights,num_rows,num_cols,bias_values = pmm.extract_weights(tflite_model_path)  

for i in range(num_layers):
    for j in zero_perch:
        save_path = "./rng_prunned_files/prunned_tflite_models/prunned_tflite_model_{}_{}.tflite".format(i,j/100)
        mf.modify_tflite_zero(tflite_model_path,save_path,[layer_indexes[i]],num_rows,num_cols,j/100)

for layer in range(0,num_layers):
    for j in zero_perch:
        modded_model = "./rng_prunned_files/prunned_tflite_models/prunned_tflite_model_{}_{}.tflite".format(layer,j/100)
        tensor_weights,num_rows,num_cols,bias_values = pmm.extract_weights(modded_model)  
        pmm.preload_layer(layer,num_rows,num_cols,tensor_weights,int(num_rows[layer]/2),num_cols[layer],shift=False,acc_threshold=j/100,DSP=1)