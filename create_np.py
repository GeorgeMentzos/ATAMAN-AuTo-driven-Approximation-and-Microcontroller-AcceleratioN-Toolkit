import os
import numpy as np

import preload_mat_mult as pmm

model_name = "LeNet_Cifar10_INT8"
model_path = "./tflite_models/" + model_name + ".tflite" 

tensor_weights,num_rows,num_cols,bias_values = pmm.extract_weights(model_path)

# splits data array to seperate b0 and b1 arrays 
def split_data(array_data):
    b0=[]
    b1=[]
    for i in range(len(array_data)):
        # if even
        if(i%2==0):
            b0.append(array_data[i])
        else:
            b1.append(array_data[i])
    return b0,b1

def create_dir(directory):
    parent_dir = os.getcwd()
    path = os.path.join(parent_dir, directory)
    os.makedirs(path, exist_ok=True)

create_dir("train_log_data_nparray")
folder_path = "./train_log_data_nparray/"
data_segs = 1000
layer_num = 3
# LeNet
mat_mult_counter = [450,84,8]

index = 1

# !!!
import output_1 as data_1
from tqdm import tqdm

total_iterations = data_segs*(layer_num-1)
progress_bar = tqdm(total=total_iterations, desc="Progress")

# converts log_file data into np arrays and saves to file
for seg in (range(0,data_segs,1)):
    # print(seg)
    segment_folder = "data_seg_{}/".format(seg)
    create_dir(folder_path+segment_folder)
    create_dir(folder_path+segment_folder)
    # skip first 2 layers
    for layer in (range(1,layer_num,1)):
        layer_folder = "layer_{}".format(layer)
        create_dir(folder_path+segment_folder+layer_folder + "/b0")
        create_dir(folder_path+segment_folder+layer_folder + "/b1")
        # array = []
        for mmc in range(0,mat_mult_counter[layer],1):
            # check which data file to import
            import_command = "array=data_{}.in{}{}{}".format(index,seg,layer,mmc+1)
            exec(import_command)
            b0,b1 = split_data(array)
            if(len(b0)!= num_cols[layer]):
                print("error at {}{}{}".format(seg,layer,mmc))
            # save b0 and b1 for each data_seg, layer, mmc
            file_path_b0 = folder_path + segment_folder + layer_folder + "/b0" + '/b0_{}{}{}.npy'.format(seg,layer,mmc+1)
            file_path_b1 = folder_path + segment_folder + layer_folder + "/b1" + '/b1_{}{}{}.npy'.format(seg,layer,mmc+1)
            np.save(file_path_b0,b0)
            np.save(file_path_b1,b1)

    if(seg%100==99):
        index+=1
        # print(index)

    progress_bar.update(1)

    import_py = "import output_{} as data_{}".format(index,index)
    exec(import_py)

# Close the progress bar when done
progress_bar.close()    