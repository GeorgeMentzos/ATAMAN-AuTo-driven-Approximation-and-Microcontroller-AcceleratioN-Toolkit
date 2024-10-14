# %%
import tensorflow as tf
from myfuncs import np2
from math import log2
import numpy as np
import os 

def create_dir(directory):
    parent_dir = os.getcwd()
    path = os.path.join(parent_dir, directory)
    os.makedirs(path, exist_ok=True)


def extract_weights(model_path,return_bias=False):
    # Load the .tflite model
    interpreter = tf.lite.Interpreter(model_path)

    op_details = interpreter._get_ops_details()
    conv_indexes = []

    for i in range(len(op_details)):
        if(op_details[i]["op_name"]=="CONV_2D"):
            conv_indexes.append(op_details[i]["index"])

    num_cols = []
    num_rows = []
    tensor_weights = [ [] for _ in range(len(conv_indexes))]
    # bias_values = [[] for _ in range(len(conv_indexes))]
    bias_values = []

    for j in range(len(conv_indexes)):
        filter_dims = interpreter._get_tensor_details(op_details[conv_indexes[j]]["inputs"][1])["shape"]
        input_shape = interpreter._get_tensor_details(op_details[conv_indexes[j]]["inputs"][0])["shape"]
        output_shape = interpreter._get_tensor_details(op_details[conv_indexes[j]]["outputs"][0])["shape"]

        # filter_dims = interpreter._get_tensor_details(op_details[conv_indexes[j]]["inputs"][1],0)["shape"]
        # input_shape = interpreter._get_tensor_details(op_details[conv_indexes[j]]["inputs"][0],0)["shape"]
        # output_shape = interpreter._get_tensor_details(op_details[conv_indexes[j]]["outputs"][0],0)["shape"]

        bias_values.append(interpreter.get_tensor(op_details[conv_indexes[j]]["inputs"][2]))
        # print(interpreter.get_tensor(op_details[conv_indexes[j]]["inputs"][2]))

        input_ch = input_shape[3]
        kernel_y = filter_dims[1]
        kernel_x = filter_dims[2]
        output_ch = output_shape[3]

        index_weights = [[] for _ in range(output_ch)]

        for k in range(output_ch):
            for l in range(kernel_y):
                for m in range(kernel_x):
                    for n in range(input_ch):
                        index_weights[k].append(interpreter.get_tensor(op_details[conv_indexes[j]]["inputs"][1])[k][l][m][n])
        
        num_cols.append(input_ch * kernel_y * kernel_x)
        num_rows.append(output_ch)
        tensor_weights[j]=index_weights

    return tensor_weights,num_rows,num_cols,bias_values

def extract_weights_np(model_path, return_bias=False):
    # Load the .tflite model
    interpreter = tf.lite.Interpreter(model_path)

    op_details = interpreter._get_ops_details()
    conv_indexes = []

    for i in range(len(op_details)):
        if op_details[i]["op_name"] == "CONV_2D":
            conv_indexes.append(op_details[i]["index"])

    num_cols = []
    num_rows = []
    tensor_weights = [np.empty(0) for _ in range(len(conv_indexes))]
    bias_values = []

    for j in range(len(conv_indexes)):
        # filter_dims = interpreter._get_tensor_details(op_details[conv_indexes[j]]["inputs"][1],0)["shape"]
        # input_shape = interpreter._get_tensor_details(op_details[conv_indexes[j]]["inputs"][0],0)["shape"]
        # output_shape = interpreter._get_tensor_details(op_details[conv_indexes[j]]["outputs"][0],0)["shape"]

        filter_dims = interpreter._get_tensor_details(op_details[conv_indexes[j]]["inputs"][1])["shape"]
        input_shape = interpreter._get_tensor_details(op_details[conv_indexes[j]]["inputs"][0])["shape"]
        output_shape = interpreter._get_tensor_details(op_details[conv_indexes[j]]["outputs"][0])["shape"]

        bias_values.append(interpreter.get_tensor(op_details[conv_indexes[j]]["inputs"][2]))

        input_ch = input_shape[3]
        kernel_y = filter_dims[1]
        kernel_x = filter_dims[2]
        output_ch = output_shape[3]

        index_weights = [[] for _ in range(output_ch)]

        for k in range(output_ch):
            for l in range(kernel_y):
                for m in range(kernel_x):
                    for n in range(input_ch):
                        # index_weights[k].append(interpreter.get_tensor(op_details[conv_indexes[j]]["inputs"][1])[k][l][m][n])
                        index_weights[k].append(interpreter.get_tensor(op_details[conv_indexes[j]]["inputs"][1],0)[k][l][m][n])

        num_cols.append(input_ch * kernel_y * kernel_x)
        num_rows.append(output_ch)
        tensor_weights[j] = np.array(index_weights)

    if return_bias:
        bias_values = np.array(bias_values)

    return tensor_weights, np.array(num_rows), np.array(num_cols), bias_values

def check_keywords(file_line,row_index=0,offset=0):
    offset_local=offset
    col_keywords = ["//DEFINITION\n","//ROW1\n","//ROW2\n","//ITERATION\n"]
    if(file_line==col_keywords[1]):
        offset_local=row_index
    if(file_line==col_keywords[2]):
        offset_local=row_index+1
    return offset_local

def write_from_file(file_path,destination,format=0,format_flag=False):
    with open(file_path,'r') as file:
        for f in file:
            if(format_flag):
                destination.write(f.format(format))
            else:    
                destination.write(f)

def pack_to_32_bit_aligned(a0, a1):
    # Sign-extend a0 and a1 to 16-bit integers
    a0_16_bit = a0 if a0 >= 0 else a0 | 0xFF00
    a1_16_bit = a1 if a1 >= 0 else a1 | 0xFF00

    # Combine the 16-bit values to form the 32-bit representation
    result = (a1_16_bit << 16) | (a0_16_bit & 0xFFFF)
    # result = (a0_16_bit << 16) | (a1_16_bit & 0xFFFF)
    return result

def pack_to_32_bit(a0, a1):
    # Sign-extend a0 and a1 to 16-bit integers
    a0_16_bit = a0 if a0 >= 0 else a0 | 0xFF00
    a1_16_bit = a1 if a1 >= 0 else a1 | 0xFF00

    # Combine the 16-bit values to form the 32-bit representation
    result = (a1_16_bit << 16) | (a0_16_bit & 0xFFFF)
    return result

def preload_weights(tensor_weights,modification_function_path,row_index,col_index,file_dest,layer):
    # choose which file to load
    with open(modification_function_path.format(0),'r') as function:
        offset=0
        for f in function:
            offset = check_keywords(f,row_index,offset)
            current_weight = tensor_weights[layer][row_index+offset][col_index]
            if(current_weight == 0):
                continue
            file_dest.write(f.format(current_weight))

def preload_weights_dsp(tensor_weights,modification_function_path,row_index,col_index,file_dest,layer,col_range):
    #NOTE THE NON CONSECUTIVE LEFTOVERS CAN BE OMITTED SEPERATLY
    if(col_index>=col_range-col_range%4):
        # print("lol")
        preload_weights(tensor_weights,modification_function_path,row_index,col_index,file_dest,layer)
    
    elif(col_index%4==0):
        with open(modification_function_path.format(6),'r') as function:
            offset=0
            # write once every 4 iterations of cols
            for f in function:
                offset = check_keywords(f,row_index,offset)
                current_weight = tensor_weights[layer][row_index+offset][col_index]
                next_current_weight =tensor_weights[layer][row_index+offset][col_index+1]
                if(current_weight==0 and next_current_weight==0):
                    continue
                packed_weight = pack_to_32_bit(current_weight,next_current_weight)
                file_dest.write(f.format(packed_weight))

        file_dest.write("dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);\n")
        file_dest.write("dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);\n\n")

        with open(modification_function_path.format(6),'r') as function:
            offset=0
            # write once every 4 iterations of cols
            for f in function:
                offset = check_keywords(f,row_index,offset)
                current_weight = tensor_weights[layer][row_index+offset][col_index+2]
                next_current_weight =tensor_weights[layer][row_index+offset][col_index+3]
                if(current_weight==0 and next_current_weight==0):
                    continue
                packed_weight = pack_to_32_bit(current_weight,next_current_weight)
                file_dest.write(f.format(packed_weight))


def preload_weights_dsp_aligned(tensor_weights,modification_function_path,row_index,col_index,file_dest,layer,col_range):
    #NOTE THE NON CONSECUTIVE LEFTOVERS CAN BE OMITTED SEPERATLY
    if(col_index>=col_range-col_range%4):
        # print("lol")
        preload_weights(tensor_weights,modification_function_path,row_index,col_index,file_dest,layer)
    
    elif(col_index%4==0):
        # smlad ordering for read and pad reordered 
        with open(modification_function_path.format(6),'r') as function:
            offset=0
            # write once every 4 iterations of cols
            for f in function:
                offset = check_keywords(f,row_index,offset)
                current_weight = tensor_weights[layer][row_index+offset][col_index]
                next_current_weight =tensor_weights[layer][row_index+offset][col_index+2]
                # if(lols_flag):
                #     print(0,current_weight,next_current_weight)
                if(current_weight==0 and next_current_weight==0):
                    continue
                packed_weight = pack_to_32_bit(current_weight,next_current_weight)
                file_dest.write(f.format(packed_weight))

        file_dest.write("dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);\n")
        file_dest.write("dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);\n\n")

        with open(modification_function_path.format(6),'r') as function:
            offset=0
            # write once every 4 iterations of cols
            for f in function:
                offset = check_keywords(f,row_index,offset)
                current_weight = tensor_weights[layer][row_index+offset][col_index+1]
                next_current_weight =tensor_weights[layer][row_index+offset][col_index+3]
                # if(lols_flag):
                #     print(1,current_weight,next_current_weight)
                if(current_weight==0 and next_current_weight==0):
                    continue
                packed_weight = pack_to_32_bit(current_weight,next_current_weight)
                file_dest.write(f.format(packed_weight))
           
def preload_odd_row_weights(cols_count,modification_function_path,file_dest,row_index,col_keywords,tensor_weights,layer):
    skip_count=0
    # fully preload final odd row for now 
    for col_index in range(cols_count):
            if(not(col_index==0)):
                write_from_file(modification_function_path.format(1),file_dest)
            with open(modification_function_path.format(0),'r') as function:
                for f in function:
                    if(f==col_keywords[2]):
                        skip_count=3
                    if(skip_count<=0):
                        current_weight = tensor_weights[layer][row_index][col_index]
                        file_dest.write(f.format(current_weight))
                    skip_count-=1

# NOTE SHIFTS ARE NOT DONE FOR THE SETS WHERE THERE IS AT LEST ONE WEIGHT = 0 ALL FOUR MULS ARE DONE NORMALLY
def preload_shift_weights(tensor_weights,modification_function_path,row_index,col_index,file_dest,layer):
    if(tensor_weights[layer][2*row_index][col_index]==0 or tensor_weights[layer][2*row_index+1][col_index]==0):
        # print("indexes",row_index,col_index)
        # print("values",tensor_weights[layer][2*row_index][col_index],tensor_weights[layer][2*row_index+1][col_index])
        preload_weights(tensor_weights,modification_function_path,row_index,col_index,file_dest,layer)
    else:
        with open(modification_function_path.format(5),'r') as function:
            offset=0
            for f in function:
                offset = check_keywords(f,row_index,offset)
                current_weight = tensor_weights[layer][row_index+offset][col_index]
                if(current_weight<0):
                    sign=-1
                else:
                    sign=1
                PoT_weight = abs(np2(current_weight))
                log_weight = int(log2(PoT_weight))
                file_dest.write(f.format(sign,log_weight))

# NOTE THIS DOES NOT ACCOUNT FOR WEIGHT=0 YET !!!FIX!!!
def preload_odd_row_shift_weights(cols_count,modification_function_path,file_dest,row_index,col_keywords,tensor_weights,layer):
    skip_count=0
    # fully preload final odd row for now 
    for col_index in range(cols_count):
            if(not(col_index==0)):
                write_from_file(modification_function_path.format(1),file_dest)
            with open(modification_function_path.format(5),'r') as function:
                for f in function:
                    if(f==col_keywords[2]):
                        skip_count=3
                    if(skip_count<=0):
                        current_weight = tensor_weights[layer][row_index][col_index]
                        if(current_weight<0):
                            sign=-1
                        else:
                            sign=1    
                        PoT_weight = abs(np2(current_weight))
                        log_weight = int(log2(PoT_weight))
                        # file_dest.write(f.format(log_weight))
                        file_dest.write(f.format(sign,log_weight))
                        # file_dest.write(f.format(current_weight))
                    skip_count-=1

# def write_weights(tensor_weights,cols_count,col_range,row_index,preload_range,modification_function_path,file_dest,shift,num_rows,layer):
#     for col_index in range(cols_count-col_range,cols_count):
#             if(not(row_index== int(num_rows[layer]/2)-preload_range and col_index==cols_count-col_range)):
#                 write_from_file(modification_function_path.format(1),file_dest,format=1,format_flag=True)  
#             if(shift):   
#                 preload_shift_weights(tensor_weights,modification_function_path,row_index,col_index,file_dest,layer)
#             else:
#                 preload_weights(tensor_weights,modification_function_path,row_index,col_index,file_dest,layer)

def write_weights(tensor_weights,cols_count,col_range,row_index,preload_range,modification_function_path,file_dest,shift,num_rows,layer,DSP):
    for col_index in range(cols_count-col_range,cols_count):
            if(DSP):
                if(not(row_index== int(num_rows[layer]/2)-preload_range and col_index==cols_count-col_range) and col_index==col_range-col_range%4):
                    # if(segment==0):
                    #     file_dest.write("int16_t b0 = *ip_b0++;\n")
                    #     file_dest.write("int16_t b1 = *ip_b1++;\n\n")
                    # else:
                        if(row_index==0):
                            file_dest.write("int16_t b0 = *ip_b0++;\n")
                            file_dest.write("int16_t b1 = *ip_b1++;\n\n")
                        else:
                            file_dest.write("b0 = *ip_b0++;\n")
                            file_dest.write("b1 = *ip_b1++;\n\n")
                elif(not(row_index== int(num_rows[layer]/2)-preload_range and col_index==cols_count-col_range) and col_index>col_range-col_range%4):
                    write_from_file(modification_function_path.format(1),file_dest,format=1,format_flag=True) 
                elif(not(row_index== int(num_rows[layer]/2)-preload_range and col_index==cols_count-col_range) and col_index%4==0):
                    file_dest.write("dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);\n")
                    file_dest.write("dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);\n\n")
            else:
                if(not(row_index== int(num_rows[layer]/2)-preload_range and col_index==cols_count-col_range)):
                    write_from_file(modification_function_path.format(1),file_dest,format=1,format_flag=True)  
            if(shift):   
                preload_shift_weights(tensor_weights,modification_function_path,row_index,col_index,file_dest,layer)
            else:
                if(DSP):
                    preload_weights_dsp(tensor_weights,modification_function_path,row_index,col_index,file_dest,layer,col_range)
                else:
                    preload_weights(tensor_weights,modification_function_path,row_index,col_index,file_dest,layer)

def write_rows(file_dest,row_index=0,segment=0):
    modification_row_function_name = "mod_row{}.c"
    parent_dir_name = "./"
    modification_row_function_path =parent_dir_name + "modifications/" + "/" + modification_row_function_name
    if(segment==0):
        write_from_file(modification_row_function_path.format(1),file_dest)
    elif(segment==1):
        write_from_file(modification_row_function_path.format(2),file_dest)
    elif(segment==2):
        # merge these!!!
        write_from_file(modification_row_function_path.format(4),file_dest)
        write_from_file(modification_row_function_path.format(3),file_dest)

# CONDITIONS CAN BE MERGED FURTHER
def write_cols(file_dest,cols_count,segment,layer,row_index,tensor_weights,preload_range,col_range,row_count,shift,num_rows,DSP):
    modification_loop_col_name = "mod_loop_col{}.c"
    parent_dir_name = "./"
    modification_loop_col_path = parent_dir_name + "modifications/" + "/" + modification_loop_col_name
    # modification files
    modification_function_name = "mod_col{}.c"
    modification_function_path = parent_dir_name + "modifications/" + "/" + modification_function_name
    col_keywords = ["//DEFINITION\n","//ROW1\n","//ROW2\n","//ITERATION\n"]
    # first time defintiions
    if(segment==0 and cols_count-col_range==0):
        if(DSP):
            file_dest.write("int32_t dsp_b0 = arm_nn_read_q15x2_ia(&ip_b0);\n")
            file_dest.write("int32_t dsp_b1 = arm_nn_read_q15x2_ia(&ip_b1);\n\n")
        else:
            write_from_file(modification_function_path.format(2),file_dest)
        # write_from_file(modification_function_path.format(2),file_dest)
    # this handles the even for loop
    elif(segment==1):
        if(cols_count-col_range!=0):
            # first time def only for the first row
            if(row_index==int(row_count/2)-preload_range):
                write_from_file(modification_function_path.format(3),file_dest)
            # iterate for all the rest
            else:
                write_from_file(modification_function_path.format(4),file_dest,format=col_range,format_flag=True)
            write_from_file(modification_loop_col_path.format(0),file_dest,format=col_range,format_flag=True)
            write_from_file(modification_loop_col_path.format(1),file_dest)
            if(row_index==int(row_count/2)-preload_range):
                write_from_file(modification_function_path.format(2),file_dest)
        write_weights(tensor_weights,cols_count,col_range,row_index,preload_range,modification_function_path,file_dest,shift,num_rows,layer,DSP=DSP)
    # this handles the remaining odd row
    elif(segment==2):
        if(shift):
            preload_odd_row_shift_weights(cols_count,modification_function_path,file_dest,row_index,col_keywords,tensor_weights,layer)
        else:
            preload_odd_row_weights(cols_count,modification_function_path,file_dest,row_index,col_keywords,tensor_weights,layer)

def preload_layer(layer,num_rows,num_cols,tensor_weights,preload_range,col_range,shift=False,acc_threshold=0,DSP=0):
    modified_folder_path = "./rng_prunned_files/"
    parent_dir_name =  "./"
    modification_loop_name = "mod_loop{}.c"
    modification_loop_path = parent_dir_name + "modifications" + "/" + modification_loop_name
    # source file
    original_file_name = "arm_nn_mat_mult_kernel_s8_s16" + ".c"
    original_kernel_path = parent_dir_name + "original_files" + "/" + original_file_name
    # destination file
    destination_file_name = "arm_nn_mat_mult_kernel_s8_s16_layer_{}".format(layer) + ".c"
    modified_folder_name = modified_folder_path + "modified_files_{}".format(acc_threshold)
    create_dir(modified_folder_name)
    modified_kernel_path = modified_folder_name + "/" + destination_file_name
    keywords_row = ["//START_PRELOAD\n","//END_PRELOAD\n","//START_PRELOAD_ODD\n","//END_PRELOAD_ODD\n","//REMAINING_PRELOAD_START\n"]

    with open(original_kernel_path,'r') as file_orig, open(modified_kernel_path,'w') as file_dest:
        flag=True
        orig_name = "int8_t *arm_nn_mat_mult_kernel_s8_s16"
        layer_name = orig_name + "_layer_{}".format(layer) + "(\n"
        skip=0
        for line in file_orig:
            # this handles renaming each layer function
            if(line=="//KERNEL_NAME\n"):
                skip = 2
                file_dest.write("//KERNEL_NAME\n")
                file_dest.write(layer_name)
            if(skip):
                skip-=1
                continue
            if(line==keywords_row[0]):
                file_dest.write(line)
                write_rows(file_dest,0)
                # placeholder row index is 0
                write_cols(file_dest,num_cols[layer],0,layer,0,tensor_weights,preload_range,col_range,num_rows[layer],shift,num_rows,DSP=DSP)
                # how many rows to preload
                for row_count in range(int(num_rows[layer]/2)-preload_range,int(num_rows[layer]/2)):
                    if(row_count!= int(num_rows[layer]/2)-preload_range):
                        write_rows(file_dest,row_count,2)    
                    write_cols(file_dest,num_cols[layer],1,layer,row_count,tensor_weights,preload_range,col_range,num_rows[layer],shift,num_rows,DSP=DSP)
                    write_rows(file_dest,num_rows[layer],1)
                flag=False
            elif(line==keywords_row[1]):
                flag=True
            # add last odd numbered row
            elif(line==keywords_row[2] and (num_rows[layer] & 0x1==1)):
                file_dest.write(line)
                write_cols(file_dest,num_cols[layer],0,layer,0,tensor_weights,preload_range,num_cols[layer],num_rows[layer],shift,num_rows,DSP=DSP)
                # HERE WE CONSIDER THAT ODD ROWS WILL ALWAYS BE PRELOADED - SUBJECT TO CHANGE
                write_cols(file_dest,num_cols[layer],2,layer,num_rows[layer]-1,tensor_weights,preload_range,col_range,num_rows[layer],shift,num_rows,DSP=DSP)
                flag=False
            elif(line==keywords_row[3] and (num_rows[layer] & 0x1==1)):
                flag=True
            if(flag):
                file_dest.write(line)
            if(line==keywords_row[4] and preload_range!=int(num_rows[layer]/2)):
                write_from_file(modification_loop_path.format(0),file_dest,format=preload_range,format_flag=True)
                write_from_file(modification_loop_path.format(1),file_dest)
