import tensorflow as tf
import numpy as np
import math
from tensorflow.lite.tools import flatbuffer_utils as fu
#import pandas as pd
from numpy import loadtxt
import random

def set_weight_value(model, layer_indexes , num_cols, exclude0, exclude1, buffers_to_skip=None):

  # Parse model buffers which store the model weights
  buffers = model.buffers
  buffer_ids = range(1, len(buffers))  # ignore index 0 as it's always None

  if buffers_to_skip is not None:
    buffer_ids = [idx for idx in buffer_ids if idx not in buffers_to_skip]

  for i in buffer_ids:
    buffer_i_data = buffers[i].data
    buffer_i_size = 0 if buffer_i_data is None else buffer_i_data.size  

    # Raw data buffers are of type ubyte (or uint8) whose values lie in the
    # range [0, 255]. Those ubytes (or unint8s) are the underlying
    # representation of each datatype. For example, a bias tensor of type
    # int32 appears as a buffer 4 times it's length of type ubyte (or uint8).
    # checks if we have our layer id in layer indexes
    if(i in layer_indexes):
      # print(num_cols)
      # print("------",layer_indexes.index(i))
      # print("--- layer {} ---".format(layer_indexes.index(i)))

      # buffer_i_size is num_rows/2 * num_cols for each layer
      # we can map the loop to the mat_mult loop using if statements
      row_index = 0
      col_index = 0
      for j in range(buffer_i_size):
        # every time we have a multiplicand of row_cound we increment the row_index
        # NOTE INGORE THE LEFTOVER ODD ROWS BY SUBTRACTING
        # all_rows = (buffer_i_size-1)// num_cols[layer_indexes.index(i)] + 1 
        # print(all_rows)
        # skip when we arrive at odd number
        row_index = j // num_cols[layer_indexes.index(i)]
        # enable this for layers that have odd rows
        # if((row_index+1)==all_rows):
        #    break
        col_index = j % num_cols[layer_indexes.index(i)]
        # val = np.int8(buffer_i_data[j])
        # print(val)
        # even numbered rows are exclude0 mapping to a0
        if(row_index%2==0):
          # print(row_index)
          # print(exclude0[layer_indexes.index(i)])
          if col_index in exclude0[layer_indexes.index(i)][row_index // 2]:
            # print("exclude0 at {},{} the value {}".format(row_index//2,col_index,val))
            # set value to 0 in tflite model
            buffer_i_data[j] = 0
        if(row_index%2!=0):
          # print(row_index)
          if col_index in exclude1[layer_indexes.index(i)][row_index // 2]:
            # print("exclude1 at {},{} the value {}".format(row_index//2,col_index,val))
            # set value to 0 in tflite model
            buffer_i_data[j] = 0

def modify_tflite_weights(layer_indexes, num_cols, exclude0, exclude1, model_path , save_path):
    # import tflite model to be modified
    model=fu.read_model_with_mutable_tensors(model_path)
    # set the weights of the model to 0 
    set_weight_value(model,layer_indexes,num_cols,exclude0,exclude1,buffers_to_skip=[1])
    fu.write_model(model,save_path)

def tflite_inference(data_path, model_path, data_samples,skip_samples=0,dataset="contest",show_output=False):
    
    if(dataset=="contest"):
      x_test = np.load(data_path + "/tmlcontest_data_npy/contestdata.npy")
      y_test = np.load(data_path + "/tmlcontest_data_npy/contestlabels.npy")

    elif(dataset=="cifar10"):
      x_test = np.load(data_path + "/cifar_data_npy/cifar_test_data.npy")
      y_test = np.load(data_path + "/cifar_data_npy/cifar_test_labels.npy")

    elif(dataset=="flowers"):
      x_test = np.load(data_path + "/flowers_data_npy/flowers_test_data.npy")
      y_test = np.load(data_path + "/flowers_data_npy/flowers_test_labels.npy")

    interpreter = tf.lite.Interpreter(model_path=model_path,num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_shape = input_details[0]['shape']
    acc = 0
    total_iter=0
    # Limit inference to the specified number of data_samples
    for i in range(skip_samples,skip_samples+data_samples):
        total_iter+=1
        input_data = x_test[i].reshape(input_shape)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        if(show_output):
          print(output_data)
        if np.argmax(output_data) == y_test[i]:
            acc += 1
    acc = acc / data_samples
    return acc

def np2(num):
  num = np.int8(num)
  sign = 1
  if(num<0):
    sign = -1
  if num == 0:
    return 0
  abs_num = abs(num)
  power = math.log2(abs_num)
  rounded_power = round(power)
  result = int(math.pow(2, rounded_power))
  return result*sign

def np2_weights(model,layer_indexes,num_rows,num_cols,col_perchentage,buffers_to_skip=None):
  # Parse model buffers which store the model weights
  buffers = model.buffers
  buffer_ids = range(1, len(buffers))  # ignore index 0 as it's always None
  if buffers_to_skip is not None:
    buffer_ids = [idx for idx in buffer_ids if idx not in buffers_to_skip]

  for i in buffer_ids:
    buffer_i_data = buffers[i].data
    buffer_i_size = 0 if buffer_i_data is None else buffer_i_data.size  

    # Raw data buffers are of type ubyte (or uint8) whose values lie in the
    # range [0, 255]. Those ubytes (or unint8s) are the underlying
    # representation of each datatype. For example, a bias tensor of type
    # int32 appears as a buffer 4 times it's length of type ubyte (or uint8).
    # TODO(b/152324470): This does not work for float as randomized weights may
    # end up as denormalized or NaN/Inf floating point numbers.
    if(i in layer_indexes):
      # maximum perchentage to convert to po2
      max_per = col_perchentage*num_cols[layer_indexes.index(i)]
      for j in range(buffer_i_size):
        # current num of column
        col_index = j % num_cols[layer_indexes.index(i)]
        if(col_index<=max_per):
          buffer_i_data[j] = np2(buffer_i_data[j])

def modify_tflite_zero(model_path,save_path,layer_indexes,num_rows,num_cols,col_perchentage):
    tflmodel = fu.read_model_with_mutable_tensors(model_path)
    zero_weights(tflmodel,layer_indexes,num_rows,num_cols,col_perchentage,buffers_to_skip=[1])
    fu.write_model(tflmodel,save_path)

def zero_weights(model,layer_indexes,num_rows,num_cols,col_perchentage,buffers_to_skip=None):
  # Parse model buffers which store the model weights
  buffers = model.buffers
  buffer_ids = range(1, len(buffers))  # ignore index 0 as it's always None
  if buffers_to_skip is not None:
    buffer_ids = [idx for idx in buffer_ids if idx not in buffers_to_skip]

  for i in buffer_ids:
    buffer_i_data = buffers[i].data
    buffer_i_size = 0 if buffer_i_data is None else buffer_i_data.size  

    # Raw data buffers are of type ubyte (or uint8) whose values lie in the
    # range [0, 255]. Those ubytes (or unint8s) are the underlying
    # representation of each datatype. For example, a bias tensor of type
    # int32 appears as a buffer 4 times it's length of type ubyte (or uint8).
    # TODO(b/152324470): This does not work for float as randomized weights may
    # end up as denormalized or NaN/Inf floating point numbers.
    if(i in layer_indexes):
      # maximum perchentage to convert to po2
      max_per = col_perchentage*num_cols[layer_indexes.index(i)]
      for j in range(buffer_i_size):
        # current num of column
        col_index = j % num_cols[layer_indexes.index(i)]
        if(col_index<=max_per):
          buffer_i_data[j] = 0 

def modify_tflite_np2(model_path,save_path,layer_indexes,num_rows,num_cols,col_perchentage):
    tflmodel = fu.read_model_with_mutable_tensors(model_path)
    np2_weights(tflmodel,layer_indexes,num_rows,num_cols,col_perchentage,buffers_to_skip=[1])
    fu.write_model(tflmodel,save_path)
    
    # with open(path+name+"_PoT.tflite", 'rb') as f:
    #   model_bytes = bytes(f.read())

    # #save the model as a c header file
    # write_header(model_bytes,c_model_name=name+"_PoT")

# Function: Convert some hex value into an array for C programming
def hex_to_c_array(hex_data, var_name):

  c_str = ''

  # Create header guard
  c_str += '#ifndef ' + var_name.upper() + '_H\n'
  c_str += '#define ' + var_name.upper() + '_H\n\n'

  # Add array length at top of file
  c_str += '\nunsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'

  # Declare C variable
  c_str += 'unsigned char ' + var_name + '[] = {'
  hex_array = []
  for i, val in enumerate(hex_data) :

    # Construct string from hex
    hex_str = format(val, '#04x')

    # Add formatting so each line stays within 80 characters
    if (i + 1) < len(hex_data):
      hex_str += ','
    if (i + 1) % 12 == 0:
      hex_str += '\n '
    hex_array.append(hex_str)

  # Add closing brace
  c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'

  # Close out header guard
  c_str += '#endif //' + var_name.upper() + '_H'

  return c_str

def representative_data_gen():

  (train_images, train_labels), (data_images, test_labels) = tf.keras.datasets.cifar10.load_data()

  # Normalize pixel values to be between 0 and 1
  train_images, data_images = train_images.astype("float32") / 255.0, data_images.astype("float32") / 255.0
#   train_name = './data_indices/train_indice.csv'
#   # Load data filenames
#   data_names = pd.read_csv(train_name)['Filename']
#   # define data array
#   x_train = [[0]*1250]*100

#   for i in range(100):
    # x_train[i] = loadtxt("./tinyml_contest_data_training/"+data_names[i]).astype("float32")
#     x_train[i] = np.reshape(x_train[i],[1, 1, 1250, 1])
#     print("progress : {0:.2f} % ".format(100*((i+1)/len(data_names))),end = '\r')

#   for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
#     yield [input_value]
    
  # Load MNIST dataset
  # mnist = tf.keras.datasets.mnist
  # (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # # input image dimensions
  # image_size = x_train.shape[1]
  # # resize and normalize
  # x_train = np.reshape(x_train,[-1, image_size, image_size, 1])
  # x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
  # x_train = x_train.astype("float32") / 255
  # x_test = x_test.astype("float32") / 255
    
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(100):
    yield [input_value]

def write_header(tflite_model,c_model_name='model'):
    # Write TFLite model to a C source (or header) file
    with open("./tflite_headers/"+c_model_name + '.h', 'w') as file:
      file.write(hex_to_c_array(tflite_model, c_model_name))
    
def convert_lite(model,name='model',quant=False,quant_type = 0,ret=False):
    # Convert the model.
    # converter = tf.lite.TFLiteConverter.from_keras_model(model+'.h5')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    quant_name = 'f32_' + name
    if(quant):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quant_name = 'int8' + name
        if(quant_type == 1):
            converter.target_spec.supported_types = [tf.float16]
            quant_name = 'f16' + name
        if(quant_type == 2):
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_data_gen
            # Ensure that if any ops can't be quantized, the converter throws an error
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # Set the input and output tensors to uint8 (APIs added in r2.3)
            # For full integer quantization, though supported types defaults to int8 only
            converter.target_spec.supported_types = [tf.int8]
            # These set the input and output tensors to uint8 (added in r2.3)
            converter.inference_input_type = tf.float32  # or tf.int8/tf.float32
            converter.inference_output_type = tf.float32 # or tf.int8/tf.float32
            # converter.inference_input_type = tf.int8  # or tf.int8/tf.float32
            # converter.inference_output_type = tf.int8 # or tf.int8/tf.float32
            quant_name = 'full_int8' + name

    tflite_model = converter.convert()

    # Save the model.
    with open("./tflite_models/"+quant_name+".tflite", 'wb') as f:
        f.write(tflite_model)
        
    #save the model as a c header file
    write_header(tflite_model,c_model_name=quant_name)
    
    if(ret):
        return tflite_model

def convert_saved_lite(model,name='model',quant=False,quant_type = 0,ret=False):
    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_saved_model(model)
    quant_name = 'f32_' + name
    if(quant):
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        quant_name = 'int8_' + name
        if(quant_type == 1):
            converter.target_spec.supported_types = [tf.float16]
            quant_name = 'f16_' + name
        if(quant_type == 2):
            converter = tf.lite.TFLiteConverter.from_saved_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = representative_data_gen
            # Ensure that if any ops can't be quantized, the converter throws an error
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            # Set the input and output tensors to uint8 (APIs added in r2.3)
            # For full integer quantization, though supported types defaults to int8 only
            converter.target_spec.supported_types = [tf.int8]
            # These set the input and output tensors to uint8 (added in r2.3)
            converter.inference_input_type = tf.float32  # or tf.int8/tf.float32
            converter.inference_output_type = tf.float32 # or tf.int8/tf.float32
            # converter.inference_input_type = tf.int8  # or tf.int8/tf.float32
            # converter.inference_output_type = tf.int8 # or tf.int8/tf.float32
            quant_name = 'full_int8' + name
    tflite_model = converter.convert()

    # Save the model.
    with open("./tflite_models/"+quant_name+".tflite", 'wb') as f:
        f.write(tflite_model)
        
    #save the model as a c header file
    write_header(tflite_model,c_model_name=quant_name)
    
    if(ret):
        return tflite_model
    
def get_weights(model_path,index):
   weight_array=[]
   tflmodel = fu.read_model_with_mutable_tensors(model_path)
   buffers = tflmodel.buffers
   buffer_i_data = buffers[index+1].data
   buffer_i_size = 0 if buffer_i_data is None else buffer_i_data.size 
   for j in range(buffer_i_size):
      weight_array.append(np.int8(buffer_i_data[j]))
   return weight_array

def cifar_data_header(data_images,data_labels,data_seg):
    data_length = data_images.shape[1]*data_images.shape[2]*data_images.shape[3]
    # data_length = data_images.shape[1]*data_images.shape[2]

    f=open("cifardata.h", "w")
    f.write("#ifndef CIFARDATA_H\n#define CIFARDATA_H\n\n")
    f.write("float cifardata[{}][{}]".format(data_seg,data_length)+"={\n")
    for i in range(data_seg):
      f.write("{")
      for row in range(data_images.shape[1]):
        for column in range(data_images.shape[2]):
            # f.write(str(data_images[i][row][column])+",")
            for color in range(data_images.shape[3]):
                f.write(str(data_images[i][row][column][color])+",")
      f.write("},\n")
    f.write("};\n\n")

    f.write("int cifarlabel[{}]".format(data_seg)+"={")
    for i in range(data_seg):
        f.write("{}".format(data_labels[i][0])+",")

    f.write("};\n\n")
    f.write("#endif")
    f.close()

# quantize input based on given scale factor
def quantization(input_value,scale_factor=0.003921568859368563,zero_point=0):
    return round(input_value / scale_factor) + zero_point

def save_cifar_data(data_images, data_labels, data_seg,name):
    path = './python_data/'
    data_length = data_images.shape[1] * data_images.shape[2] * data_images.shape[3]
    cifar_data = np.zeros((data_seg, data_length), dtype=np.float32)
    quant_cifar_data = np.zeros((data_seg, data_length), dtype=np.uint8)
    for i in range(data_seg):
        for row in range(data_images.shape[1]):
            for column in range(data_images.shape[2]):
                for color in range(data_images.shape[3]):
                    cifar_data[i][row * data_images.shape[2] * data_images.shape[3]
                                + column * data_images.shape[3] + color] = data_images[i][row][column][color]
                    quant_cifar_data[i][row * data_images.shape[2] * data_images.shape[3]
                                + column * data_images.shape[3] + color] = quantization(float(data_images[i][row][column][color]))

    cifar_labels = np.array(data_labels[:data_seg], dtype=np.int32)

    np.save(path+name+"_data.npy", cifar_data)
    np.save(path+name+"_quant_data.npy", quant_cifar_data)
    np.save(path+name+"_labels.npy", cifar_labels)


# def tflite_inference(data_path,model_path,data_samples):
#     # x_test = np.load(data_path+"/cifar_data_npy/tinyml_data.npy")
#     # y_test = np.load(data_path+"/cifar_data_npy/cifar_test_labels.npy")

#     x_test = np.load(data_path+"/tmlcontest_data_npy/contestdata.npy")
#     y_test = np.load(data_path+"/tmlcontest_data_npy/contestlabels.npy")



#     interpreter = tf.lite.Interpreter(model_path=model_path)
#     interpreter.allocate_tensors()
#     # Get input and output tensors.
#     input_details = interpreter.get_input_details()
#     output_details = interpreter.get_output_details()
#     # Test model on some input data.
#     input_shape = input_details[0]['shape']
#     acc=0

#     # skip first 2000 samples
#     x_test = x_test[2000:data_samples+1000]

#     for i in range(len(x_test)):
#         input_data = x_test[i].reshape(input_shape)
#         interpreter.set_tensor(input_details[0]['index'], input_data)
#         interpreter.invoke()
#         output_data = interpreter.get_tensor(output_details[0]['index'])
#         # np.argmax if using data from python 
#         # use actual index for imported user data
#         # print(output_data)
#         if(np.argmax(output_data) == y_test[i+2000]):
#             acc+=1
#     acc = acc/len(x_test)
#     # print('test accuracy is : {} %'.format(acc*100))
#     return acc
