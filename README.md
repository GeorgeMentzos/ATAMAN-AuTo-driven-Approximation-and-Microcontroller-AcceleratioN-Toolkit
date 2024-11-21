# ATAMAN: AuTo-driven Approximation and Microcontroller AcceleratioN Toolkit

To run an example for a TensorFlow Lite model the run_all.sh script can be used. 

1) calculate_significance.py takes as input the desired start, stop and step for calculating the significance of each threshold value within the used defined range.

2) generate_tflit.py take as input the output significance for the model and set all weights below the threshold to 0 and then it as a new .tflite model

3) dse.py performs exhaustive design space exploration by testing all .tflite models to calculate the classification accuracy and saves the results.

4) plot_generation.py generates the pareto plot for the input .tflite model

5) unpack.py accepts an additional argument (maximum accuracy loss), which chooses the best configuration from the pareto optimal models that satisfies the accuracy constraint. The desired matrix multiplication kernel header files are then generated, which can be imported to the tflite micro project and flashed to the microcontroller.
