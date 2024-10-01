from preload_mat_mult import extract_weights_np
import numpy as np
import os
import time
import multiprocessing
from tqdm import tqdm

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

# NOTE THIS DOES NOT WORK BECAUSE EXEC DOES NOT SAVE LOCALLY TO FUNCTION
# EXCECUTION MUST BE DONE OUTSIDE OF FUNCTION
def save_np():
    create_dir("train_log_data_nparray")
    folder_path = "./train_log_data_nparray/"
    data_segs = 20
    layer_num = 3
    # NOTE  generalize for the rest general formula out_x*out_y/2
    mat_mult_counter = [450,84,8]

    # converts log_file data into np arrays and saves to file
    for seg in (range(0,data_segs,1)):
        segment_folder = "data_seg_{}/".format(seg)
        create_dir(folder_path+segment_folder)
        create_dir(folder_path+segment_folder)
        for layer in (range(0,layer_num,1)):
            layer_folder = "layer_{}".format(layer)
            create_dir(folder_path+segment_folder+layer_folder + "/b0")
            create_dir(folder_path+segment_folder+layer_folder + "/b1")
            for mmc in range(0,mat_mult_counter[layer],1):
                array=[]
                import_command = "array=data.in{}{}{}".format(seg,layer,mmc+1)
                exec(import_command)
                b0,b1 = split_data(array)
                # save b0 and b1 for each data_seg, layer, mmc
                file_path_b0 = folder_path + segment_folder + layer_folder + "/b0" + '/b0_{}{}{}.npy'.format(seg,layer,mmc+1)
                file_path_b1 = folder_path + segment_folder + layer_folder + "/b1" + '/b1_{}{}{}.npy'.format(seg,layer,mmc+1)
                np.save(file_path_b0,b0)
                np.save(file_path_b1,b1)

# create the partial products and split to channels and saves to np arrays
# this function creates partial products for 1 iteration of mat_mult_kernel
def save_pp_channels_np(mmc, layer, data_seg,tensor_weights, num_rows, num_cols, bias_values):
    # init ch arrays containing num_row * num_col partial products
    # all partial products for one row are computed using the same b values and
    # weight values differ from row to row
    ar00 = np.empty((int(num_rows[layer]/2), num_cols[layer]), dtype=np.float32)
    ar01 = np.empty((int(num_rows[layer]/2), num_cols[layer]), dtype=np.float32)
    ar10 = np.empty((int(num_rows[layer]/2), num_cols[layer]), dtype=np.float32)
    ar11 = np.empty((int(num_rows[layer]/2), num_cols[layer]), dtype=np.float32)

    # Create directory for each data segment
    create_dir("./partial_products/data_seg_{}".format(data_seg))
    # Create directory for each layer
    create_dir("./partial_products/data_seg_{}/layer_{}".format(data_seg,layer))

    # Create directory for each arr
    create_dir("./partial_products/data_seg_{}/layer_{}/ar00".format(data_seg,layer))
    create_dir("./partial_products/data_seg_{}/layer_{}/ar01".format(data_seg,layer))
    create_dir("./partial_products/data_seg_{}/layer_{}/ar10".format(data_seg,layer))
    create_dir("./partial_products/data_seg_{}/layer_{}/ar11".format(data_seg,layer))
    
    file_path_ar00 = "./partial_products/data_seg_{}/layer_{}/ar00/ar00_{}{}{}.npy".format(data_seg,layer, data_seg,layer,mmc+1)
    file_path_ar01 = "./partial_products/data_seg_{}/layer_{}/ar01/ar01_{}{}{}.npy".format(data_seg,layer, data_seg,layer,mmc+1)
    file_path_ar10 = "./partial_products/data_seg_{}/layer_{}/ar10/ar10_{}{}{}.npy".format(data_seg,layer, data_seg,layer,mmc+1)
    file_path_ar11 = "./partial_products/data_seg_{}/layer_{}/ar11/ar11_{}{}{}.npy".format(data_seg,layer, data_seg,layer,mmc+1)

    # generate the b arrays containing the corresponding input values
    # for each layer, and column
    folder_path0 = "./ver_train_log_data_nparray/data_seg_{}/layer_{}/b0".format(data_seg, layer)
    folder_path1 = "./ver_train_log_data_nparray/data_seg_{}/layer_{}/b1".format(data_seg, layer)
    file_name_b0 = "/b{}_{}{}{}.npy".format(0, data_seg, layer, mmc+1)
    file_name_b1 = "/b{}_{}{}{}.npy".format(1, data_seg, layer, mmc+1)
    # arrays with num_cols values each
    b0 = np.load(folder_path0 + file_name_b0)
    b1 = np.load(folder_path1 + file_name_b1)
    print(data_seg,layer,mmc+1)

    # for j in range(0, int(num_rows[layer]/2)):
    #     offset = j
    #     bias0 = bias_values[layer][j+offset]
    #     bias1 = bias_values[layer][j+1+offset]
    #     # generate a0 a1 for rach row with num col elements
    #     a0 = tensor_weights[layer][j+offset]
    #     a1 = tensor_weights[layer][j+1+offset]

    #     # do element wise multiplication initilize with zeroes in the for of our arrays
    #     ar00 = np.zeros(len(a0))
    #     ar01 = np.zeros(len(a0))
    #     ar10 = np.zeros(len(a0))
    #     ar11 = np.zeros(len(a0))
    #     # initizalize with bias
    #     ar00[0] = bias0 + a0[0]*b0[0]
    #     ar01[0] = bias0 + a0[0]*b1[0]
    #     ar10[0] = bias1 + a1[0]*b0[0]
    #     ar11[0] = bias1 + a1[0]*b1[0]
        
    #     # Perform element-wise multiplication and cumulative sum
    #     ar00[1:] = np.cumsum(a0[1:] * b0[1:]) + ar00[0]
    #     ar01[1:] = np.cumsum(a0[1:] * b1[1:]) + ar01[0]
    #     ar10[1:] = np.cumsum(a1[1:] * b0[1:]) + ar10[0]
    #     ar11[1:] = np.cumsum(a1[1:] * b1[1:]) + ar11[0]

    for j in range(0, int(num_rows[layer]/2)):
        offset = j
        ch00 = ch01 = ch10 = ch11 = 0
        ch00 = ch01 = bias_values[layer][j+offset]
        ch10 = ch11 = bias_values[layer][j+1+offset]

        for i in range(0, num_cols[layer]):
            # generate the a arrays containing the corresponding weights
            # for each layer, row and column
            a0 = tensor_weights[layer][j+offset][i]
            a1 = tensor_weights[layer][j+1+offset][i]

            # ch00 += a0 * b0[i]
            # ar00[j][i] = ch00
            # ch01 += a0 * b1[i]
            # ar01[j][i] = ch01
            # ch10 += a1 * b0[i]
            # ar10[j][i] = ch10
            # ch11 += a1 * b1[i]
            # ar11[j][i] = ch11

            ch00 = a0 * b0[i]
            ar00[j][i] = ch00
            ch01 = a0 * b1[i]
            ar01[j][i] = ch01
            ch10 = a1 * b0[i]
            ar10[j][i] = ch10
            ch11 = a1 * b1[i]
            ar11[j][i] = ch11

    # Save the partial products to separate files
    np.save(file_path_ar00, ar00)
    np.save(file_path_ar01, ar01)
    np.save(file_path_ar10, ar10)
    np.save(file_path_ar11, ar11)

    return ar00, ar01, ar10, ar11

# create the partial products and split to channels and saves to np arrays
# this function creates partial products for 1 iteration of mat_mult_kernel
def pp_channels_np(mmc, layer, data_seg,tensor_weights, num_rows, num_cols, bias_values):
    # init ch arrays containing num_row * num_col partial products
    # all partial products for one row are computed using the same b values and
    # weight values differ from row to row
    ar00 = np.empty((int(num_rows[layer]/2), num_cols[layer]), dtype=np.float32)
    ar01 = np.empty((int(num_rows[layer]/2), num_cols[layer]), dtype=np.float32)
    ar10 = np.empty((int(num_rows[layer]/2), num_cols[layer]), dtype=np.float32)
    ar11 = np.empty((int(num_rows[layer]/2), num_cols[layer]), dtype=np.float32)

    for j in range(0, int(num_rows[layer]/2)):
        offset = j
        ch00 = ch01 = ch10 = ch11 = 0
        ch00 = ch01 = bias_values[layer][j+offset]
        ch10 = ch11 = bias_values[layer][j+1+offset]

        # generate the b arrays containing the corresponding input values
        # for each layer, and column
        folder_path0 = "./train_log_data_nparray/data_seg_{}/layer_{}/b0".format(data_seg, layer)
        folder_path1 = "./train_log_data_nparray/data_seg_{}/layer_{}/b1".format(data_seg, layer)
        file_name_b0 = "/b{}_{}{}{}.npy".format(0, data_seg, layer, mmc+1)
        file_name_b1 = "/b{}_{}{}{}.npy".format(1, data_seg, layer, mmc+1)
        # file_name_b0 = "/b{}_{}{}{}.npy".format(0, data_seg, layer, mmc)
        # file_name_b1 = "/b{}_{}{}{}.npy".format(1, data_seg, layer, mmc)
        # arrays with num_cols values each
        b0 = np.load(folder_path0 + file_name_b0)
        b1 = np.load(folder_path1 + file_name_b1)

        for i in range(0, num_cols[layer]):
            # generate the a arrays containing the corresponding weights
            # for each layer, row and column
            a0 = tensor_weights[layer][j+offset][i]
            a1 = tensor_weights[layer][j+1+offset][i]

            ch00 += a0 * b0[i]
            ar00[j][i] = ch00
            ch01 += a0 * b1[i]
            ar01[j][i] = ch01
            ch10 += a1 * b0[i]
            ar10[j][i] = ch10
            ch11 += a1 * b1[i]
            ar11[j][i] = ch11

    return ar00, ar01, ar10, ar11

def distance_metrics_np(layer, data_seg, mmc, tensor_weights, num_rows, num_cols, bias_values):
    ar00, ar01, ar10, ar11 = pp_channels_np(mmc, layer, data_seg, tensor_weights, num_rows, num_cols, bias_values)
    
    # ar00 = np.load("./partial_products/data_seg_{}/layer_{}/ar00/ar00_{}{}{}.npy".format(data_seg,layer, data_seg,layer,mmc+1))
    # ar01 = np.load("./partial_products/data_seg_{}/layer_{}/ar01/ar01_{}{}{}.npy".format(data_seg,layer, data_seg,layer,mmc+1))
    # ar10 = np.load("./partial_products/data_seg_{}/layer_{}/ar10/ar10_{}{}{}.npy".format(data_seg,layer, data_seg,layer,mmc+1))
    # ar11 = np.load("./partial_products/data_seg_{}/layer_{}/ar11/ar11_{}{}{}.npy".format(data_seg,layer, data_seg,layer,mmc+1))
    
    dar00 = np.empty((int(num_rows[layer]/2), num_cols[layer]), dtype=np.float32)
    dar01 = np.empty((int(num_rows[layer]/2), num_cols[layer]), dtype=np.float32)
    dar10 = np.empty((int(num_rows[layer]/2), num_cols[layer]), dtype=np.float32)
    dar11 = np.empty((int(num_rows[layer]/2), num_cols[layer]), dtype=np.float32)

    for j in range(0, int(num_rows[layer]/2)):
        group = 0
        skips = 0

        # calculate sum for each array
        target_value00 = np.sum(ar00[j])
        target_value01 = np.sum(ar01[j])
        target_value10 = np.sum(ar10[j])
        target_value11 = np.sum(ar11[j])

        for i in range(0, num_cols[layer]):
            dar00[j][i] = abs(ar00[j][i]/target_value00)
            dar01[j][i] = abs(ar01[j][i]/target_value01)
            dar10[j][i] = abs(ar10[j][i]/target_value10)
            dar11[j][i] = abs(ar11[j][i]/target_value11)

    return dar00, dar01, dar10, dar11

# divide each element of the array by total samples
def array2d_mean_np(array, samples):
    array /= samples
    return array

# this function is the same for all runs no reason to run multiple times
# calculates mean distance metric for all data segs and all mat mult iterations
def mean_distance_metric_np(layer, data_len, tensor_weights, num_rows, num_cols, bias_values):
    mean_dar00 = np.zeros((int(num_rows[layer]/2), num_cols[layer]))
    mean_dar01 = np.zeros((int(num_rows[layer]/2), num_cols[layer]))
    mean_dar10 = np.zeros((int(num_rows[layer]/2), num_cols[layer]))
    mean_dar11 = np.zeros((int(num_rows[layer]/2), num_cols[layer]))

    # NOTE PARSE THIS AS ARGUMENT TO GENERALIZE
    # mat_mult_counter = [450, 84, 8]
    # mat_mult_counter = [623//2,310//2,154//2,76//2,37//2]
    # mat_mult_counter = [512,0,0,512,0,0,128,0,0,32,0,0,8,0,0,2]
    # LeNet
    # mat_mult_counter = [450,84,8]
    # AlexNet
    mat_mult_counter = [(30*30)//2,(13*13)//2,(11*11)//2,(9*9)//2,(2*2)//2]

    # 3 mat_mult counter will be considered
    total_iterations = (data_len-10) * mat_mult_counter[layer]
    progress_bar = tqdm(total=total_iterations, desc="Progress")

    # NOTE SKIP THE FIRST 10 SAMPLES OF DATA SET 990 TOTAL
    for data_seg in range(11,data_len,1):
        for mmc in range(mat_mult_counter[layer]):
            # print(data_seg, mmc)
            dar00, dar01, dar10, dar11 = distance_metrics_np(layer, data_seg, mmc, tensor_weights, num_rows, num_cols, bias_values)
            mean_dar00 += np.array(dar00)
            mean_dar01 += np.array(dar01)
            mean_dar10 += np.array(dar10)
            mean_dar11 += np.array(dar11)
            progress_bar.update(1)

    # Close the progress bar when done
    progress_bar.close()

    mean_dar00 = array2d_mean_np(mean_dar00, data_len*mat_mult_counter[layer])
    mean_dar01 = array2d_mean_np(mean_dar01, data_len*mat_mult_counter[layer])
    mean_dar10 = array2d_mean_np(mean_dar10, data_len*mat_mult_counter[layer])
    mean_dar11 = array2d_mean_np(mean_dar11, data_len*mat_mult_counter[layer])

    dir_name = "./mean_values_np_array_{}_samples".format(data_len)
    create_dir(dir_name)
    layer_dir_name = "/layer_{}".format(layer)
    create_dir(dir_name+layer_dir_name)

    file_path = dir_name + layer_dir_name + "/mean_dar{}{}_layer{}.npy"

    np.save(file_path.format(0,0,layer),mean_dar00)
    np.save(file_path.format(0,1,layer),mean_dar01)
    np.save(file_path.format(1,0,layer),mean_dar10)
    np.save(file_path.format(1,1,layer),mean_dar11)

    # print("mean values saved for layer {}".format(layer))  

    return mean_dar00, mean_dar01, mean_dar10, mean_dar11

# creates list of consecutive indexes
def cons2(indices_list):
    consecutive_or_multiples_of_two_indices = []
    current_consecutive = []
    for index in indices_list:
        if not current_consecutive or index == current_consecutive[-1] + 1:
            current_consecutive.append(index)
        else:
            current_consecutive = [index]
        if len(current_consecutive) >= 2 and len(current_consecutive) % 2 == 0:
            consecutive_or_multiples_of_two_indices.extend(current_consecutive)
            current_consecutive = []  # Reset the current_consecutive list after adding
    return consecutive_or_multiples_of_two_indices

# maps the consecutive index list to new indexes to be used by preloading script for dsp omission 
def create_2_value_combinations(indices_list):
    return list(range(len(indices_list) // 2))

def significance(layer, data_len, threshold0, threshold1, num_rows, num_cols, global_mean_data_path, path,skip_layer=2):

    # Save the common values to a Python header file
    with open(path, "w") as f:
        f.write("#significance_indexes\n")
        f.write("#threshold0={}\n".format(threshold0))
        f.write("#threshold1={}\n".format(threshold1))

    # skip first 2 layer in all computations 
    for layer_index in range(skip_layer,layer):
        # mean_dar00,mean_dar01,mean_dar10,mean_dar11 = mean_distance_metric_np(layer_index,data_len)

        # import means from a file to avoid compute overhead for each iter
        dir_name = global_mean_data_path  + "/mean_values_np_array_{}_samples".format(data_len)
        layer_dir_name = "/layer_{}".format(layer_index)

        file_path = dir_name + layer_dir_name + "/mean_dar{}{}_layer{}.npy"

        mean_dar00 = np.load(file_path.format(0,0,layer_index))
        mean_dar01 = np.load(file_path.format(0,1,layer_index))
        mean_dar10 = np.load(file_path.format(1,0,layer_index))
        mean_dar11 = np.load(file_path.format(1,1,layer_index))

        values_below_threshold00 = []
        values_below_threshold01 = []
        values_below_threshold10 = []
        values_below_threshold11 = []

        common_values_below_threshold0 = []
        common_values_below_threshold1 = []

        consecutive_common_values_below_threshold0 = []
        consecutive_common_values_below_threshold1 = []

        exclude_dsp0 = []
        exclude_dsp1 = []

        perforated_macs = 0 
        perforated_macs_dsp = 0
        # num rows/2 * num cols * mac per iter(4)
        total_macs = (int(num_rows[layer_index]/2))*num_cols[layer_index]*4
        
        for row in range(int(num_rows[layer_index] / 2)):
            values_below_threshold00.append([index for index, value in enumerate(mean_dar00[row]) if (value < threshold0[layer_index]) and (value != 0.0)])
            values_below_threshold01.append([index for index, value in enumerate(mean_dar01[row]) if (value < threshold0[layer_index]) and (value != 0.0)])
            values_below_threshold10.append([index for index, value in enumerate(mean_dar10[row]) if (value < threshold1[layer_index]) and (value != 0.0)])
            values_below_threshold11.append([index for index, value in enumerate(mean_dar11[row]) if (value < threshold1[layer_index]) and (value != 0.0)])

            common_values_below_threshold0.append(list(set(values_below_threshold00[row]).intersection(values_below_threshold01[row])))
            common_values_below_threshold1.append(list(set(values_below_threshold10[row]).intersection(values_below_threshold11[row])))

            consecutive_common_values_below_threshold0.append(cons2(list(set(values_below_threshold00[row]).intersection(values_below_threshold01[row]))))
            consecutive_common_values_below_threshold1.append(cons2(list(set(values_below_threshold10[row]).intersection(values_below_threshold11[row]))))

        # print("hello")
        for k in range(int(num_rows[layer_index]/2)):
            if(len(consecutive_common_values_below_threshold0[k])%2!=0 or len(consecutive_common_values_below_threshold1[k])%2!=0):
                print("error")
            # remap consecutive values to arrays with 1/4 the length
            exclude_dsp0.append(create_2_value_combinations(consecutive_common_values_below_threshold0[k]))
            exclude_dsp1.append(create_2_value_combinations(consecutive_common_values_below_threshold1[k]))
            # calculate perforation perchentage
            perforated_macs+= 2*(len(common_values_below_threshold0[k]) + len(common_values_below_threshold1[k]))
            perforated_macs_dsp+= 2*(len(consecutive_common_values_below_threshold0[k]) + len(consecutive_common_values_below_threshold1[k]))

        perforation_perchentage = perforated_macs/total_macs*100
        perforation_perchentage_dsp = perforated_macs_dsp/total_macs*100

        # Save the common values to a Python header file
        with open(path, "a") as f:
            f.write("layer{}_values_below_threshold00 = {}\n".format(layer_index,values_below_threshold00))
            f.write("layer{}_values_below_threshold01 = {}\n".format(layer_index,values_below_threshold01))
            f.write("layer{}_values_below_threshold10 = {}\n".format(layer_index,values_below_threshold10))
            f.write("layer{}_values_below_threshold11 = {}\n".format(layer_index,values_below_threshold11))
            f.write("layer{}_consecutive_common_values_below_threshold0 = {}\n".format(layer_index,consecutive_common_values_below_threshold0))
            f.write("layer{}_consecutive_common_values_below_threshold1 = {}\n".format(layer_index,consecutive_common_values_below_threshold1))
            f.write("layer{}_common_values_below_threshold0 = {}\n".format(layer_index,common_values_below_threshold0))
            f.write("layer{}_common_values_below_threshold1 = {}\n".format(layer_index,common_values_below_threshold1))
            f.write("layer{}_exclude_dsp0 = {}\n".format(layer_index,exclude_dsp0))
            f.write("layer{}_exclude_dsp1 = {}\n".format(layer_index,exclude_dsp1))
            f.write("################\n")
            f.write("layer_{}_total_macs = {}\n".format(layer_index,total_macs))
            f.write("layer_{}_perforated_macs = {}\n".format(layer_index,perforated_macs))
            f.write("layer_{}_perforated_macs_dsp = {}\n".format(layer_index,perforated_macs_dsp))
            f.write("layer_{}_perforation_perchentage = {}\n".format(layer_index,perforation_perchentage))
            f.write("layer_{}_perforation_perchentage_dsp = {}\n".format(layer_index,perforation_perchentage_dsp))
            f.write("################\n")


# def significance_v2(layer, data_len, threshold0, threshold1, num_rows, num_cols, global_mean_data_path, path):

#     # Save the common values to a Python header file
#     with open(path, "w") as f:
#         f.write("#significance_indexes\n")
#         f.write("#threshold0={}\n".format(threshold0))
#         f.write("#threshold1={}\n".format(threshold1))

#     # skip first 2 layer in all computations 
#     for layer_index in layer:
#         # mean_dar00,mean_dar01,mean_dar10,mean_dar11 = mean_distance_metric_np(layer_index,data_len)

#         # import means from a file to avoid compute overhead for each iter
#         dir_name = global_mean_data_path  + "/mean_values_np_array_{}_samples".format(data_len)
#         layer_dir_name = "/layer_{}".format(layer_index)

#         file_path = dir_name + layer_dir_name + "/mean_dar{}{}_layer{}.npy"

#         mean_dar00 = np.load(file_path.format(0,0,layer_index))
#         mean_dar01 = np.load(file_path.format(0,1,layer_index))
#         mean_dar10 = np.load(file_path.format(1,0,layer_index))
#         mean_dar11 = np.load(file_path.format(1,1,layer_index))

#         values_below_threshold00 = []
#         values_below_threshold01 = []
#         values_below_threshold10 = []
#         values_below_threshold11 = []

#         common_values_below_threshold0 = []
#         common_values_below_threshold1 = []

#         consecutive_common_values_below_threshold0 = []
#         consecutive_common_values_below_threshold1 = []

#         exclude_dsp0 = []
#         exclude_dsp1 = []

#         perforated_macs = 0 
#         perforated_macs_dsp = 0
#         # num rows/2 * num cols * mac per iter(4)
#         total_macs = (int(num_rows[layer_index]/2))*num_cols[layer_index]*4
        
#         for row in range(int(num_rows[layer_index] / 2)):
#             values_below_threshold00.append([index for index, value in enumerate(mean_dar00[row]) if (value < threshold0[layer_index]) and (value != 0.0)])
#             values_below_threshold01.append([index for index, value in enumerate(mean_dar01[row]) if (value < threshold0[layer_index]) and (value != 0.0)])
#             values_below_threshold10.append([index for index, value in enumerate(mean_dar10[row]) if (value < threshold1[layer_index]) and (value != 0.0)])
#             values_below_threshold11.append([index for index, value in enumerate(mean_dar11[row]) if (value < threshold1[layer_index]) and (value != 0.0)])

#             common_values_below_threshold0.append(list(set(values_below_threshold00[row]).intersection(values_below_threshold01[row])))
#             common_values_below_threshold1.append(list(set(values_below_threshold10[row]).intersection(values_below_threshold11[row])))

#             consecutive_common_values_below_threshold0.append(cons2(list(set(values_below_threshold00[row]).intersection(values_below_threshold01[row]))))
#             consecutive_common_values_below_threshold1.append(cons2(list(set(values_below_threshold10[row]).intersection(values_below_threshold11[row]))))

#         #     for i in range(len(values_below_threshold00[row]) - 1):
#         #         pair00 = [values_below_threshold00[row][i], values_below_threshold00[row][i + 1]]
#         #         pair01 = [values_below_threshold01[row][i], values_below_threshold01[row][i + 1]]
                
#         #         common_values0 = list(set(pair00).intersection(pair01))                
#         #         consecutive_common_values_below_threshold0.append(common_values)

#         #     for i in range(len(values_below_threshold10[row]) - 1):
#         #         pair10 = [values_below_threshold10[row][i], values_below_threshold10[row][i + 1]]
#         #         pair11 = [values_below_threshold11[row][i], values_below_threshold11[row][i + 1]]
                
#         #         common_values0 = list(set(pair10).intersection(pair11))
#         #         consecutive_common_values_below_threshold1.append(common_values)

#         # for k in range(int(num_rows[layer_index]/2)):
#         #     if(len(consecutive_common_values_below_threshold0[k])%2!=0 or len(consecutive_common_values_below_threshold1[k])%2!=0):
#         #         print("error")

#             # remap consecutive values to arrays with 1/2 the length
#             exclude_dsp0.append(create_2_value_combinations(consecutive_common_values_below_threshold0[k]))
#             exclude_dsp1.append(create_2_value_combinations(consecutive_common_values_below_threshold1[k]))
#             # calculate perforation perchentage
#             perforated_macs+= len(common_values_below_threshold0[k]) + len(common_values_below_threshold1[k])
#             perforated_macs_dsp+= len(consecutive_common_values_below_threshold0[k]) + len(consecutive_common_values_below_threshold1[k])

#         perforation_perchentage = perforated_macs/total_macs*100
#         perforation_perchentage_dsp = perforated_macs_dsp/total_macs*100

#         # Save the common values to a Python header file
#         with open(path, "a") as f:
#             f.write("layer{}_values_below_threshold00 = {}\n".format(layer_index,values_below_threshold00))
#             f.write("layer{}_values_below_threshold01 = {}\n".format(layer_index,values_below_threshold01))
#             f.write("layer{}_values_below_threshold10 = {}\n".format(layer_index,values_below_threshold10))
#             f.write("layer{}_values_below_threshold11 = {}\n".format(layer_index,values_below_threshold11))
#             f.write("layer{}_consecutive_common_values_below_threshold0 = {}\n".format(layer_index,consecutive_common_values_below_threshold0))
#             f.write("layer{}_consecutive_common_values_below_threshold1 = {}\n".format(layer_index,consecutive_common_values_below_threshold1))
#             f.write("layer{}_common_values_below_threshold0 = {}\n".format(layer_index,common_values_below_threshold0))
#             f.write("layer{}_common_values_below_threshold1 = {}\n".format(layer_index,common_values_below_threshold1))
#             f.write("layer{}_exclude_dsp0 = {}\n".format(layer_index,exclude_dsp0))
#             f.write("layer{}_exclude_dsp1 = {}\n".format(layer_index,exclude_dsp1))
#             f.write("################\n")
#             f.write("layer_{}_total_macs = {}\n".format(layer_index,total_macs))
#             f.write("layer_{}_perforated_macs = {}\n".format(layer_index,perforated_macs))
#             f.write("layer_{}_perforated_macs_dsp = {}\n".format(layer_index,perforated_macs_dsp))
#             f.write("layer_{}_perforation_perchentage = {}\n".format(layer_index,perforation_perchentage))
#             f.write("layer_{}_perforation_perchentage_dsp = {}\n".format(layer_index,perforation_perchentage_dsp))
#             f.write("################\n")
