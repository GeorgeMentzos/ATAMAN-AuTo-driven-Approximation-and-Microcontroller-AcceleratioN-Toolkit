import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os 
import shutil

WIDTH_SIZE=3.49
HEIGHT_SIZE=2
MARKERSIZE=10
ALPHA_MIN = 0.1  # Minimum alpha value (lighter)
ALPHA_MAX = 1.0  # Maximum alpha value (darker)

def create_dir(directory):
    parent_dir = os.getcwd()
    path = os.path.join(parent_dir, directory)
    os.makedirs(path, exist_ok=True)


def is_dominated(candidate, solutions):
    return np.any(np.all(solutions >= candidate, axis=1))
    
# def is_dominated(candidate, solutions):
#     return np.any(np.all(solutions <= candidate, axis=1)) and np.any(np.any(solutions < candidate, axis=1))

def find_pareto_optimal(data):
    pareto_optimal = []
    for i, candidate in enumerate(data):
        if not is_dominated(candidate, np.delete(data, i, axis=0)):
            pareto_optimal.append(candidate)
    return np.array(pareto_optimal)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process start and stop arguments.")
    parser.add_argument("start", type=int, help="Start value for the range")
    parser.add_argument("stop", type=int, help="Stop value for the range")
    parser.add_argument("step", type=int, nargs="?", default=1, help="Step value for the range")
    parser.add_argument("acc_threshold", type=float, help="Accuracy threshold value")
    args = parser.parse_args()

    start = args.start
    stop = args.stop
    step = args.step
    acc_threshold = args.acc_threshold

    dse_res_folder = "dse_res_{}_{}_{}".format(start/1000,stop/1000,step)
    dse_res_folder_path = "./" + dse_res_folder + "/"

    # import from dse reults acc and perforation values
    x_values = np.load(dse_res_folder_path + "x_vals_{}_{}_{}.npy".format(start,stop,step))
    y_values = np.load(dse_res_folder_path + "y_vals_{}_{}_{}.npy".format(start,stop,step))

    # stack for finding pareto optimal 
    data = np.column_stack((x_values, y_values))
    # print(data)
    pareto_optimal_solutions = find_pareto_optimal(data)

    # calculate a max acc threshold total acc loss from original
    orig_acc = y_values[0]
    minimum_acc = orig_acc-acc_threshold

    layers=3
    skip_layers=1
    factor=1
    model_name = "LeNet_Cifar10_INT8.tflite"

    total_iterations = ((stop - start)//step) ** (layers-skip_layers)
    progress_bar = tqdm(total=total_iterations, desc="Progress")

    count = 0
    best_x = 0
    best_y=0
    best_threshold=[-10,0.0,0.0]
    # best_threshold=[-10,0.0,0.0,0.0,0.0]

    for i in [float(ii) / 1000 for ii in range(start, stop, step)]:
        for j in [float(jj) / 1000 for jj in range(start, stop, step)]:
            # for k in [float(kk) / 1000 for kk in range(start, stop, step)]:
            #     for l in [float(ll) / 1000 for ll in range(start, stop, step)]:
                        # import the neccesarry info from .tflite file 
                        threshold = [-10,i*factor,j]
                        # up to 6% acc loss
                        if(y_values[count]>=minimum_acc):
                            if(x_values[count]>best_x):
                                # add cache penalty condition
                                # if(x_values[count]> 0.15):
                                    # best speedup for acc loss up to 2%
                                best_x = x_values[count]
                                best_y = y_values[count]
                                best_threshold = threshold
                        count+=1
                        progress_bar.update(1)

    # Close the progress bar when done
    progress_bar.close()

    # file destination paths
    plot_folder = "plot_{}_{}_{}".format(start/1000,stop/1000,step)
    create_dir(plot_folder)
    plot_destination = "./" + plot_folder + "/dse_{}_{}_{}_{}.pdf"
    pareto_plot_destination = "./" + plot_folder + "/pareto_dse_{}_{}_{}_{}.pdf"
    siginificance_folder_name = "significance_folder_{}_{}_{}".format(start/1000,(stop-1)/1000,step)
        # significance_index_file_path = "./" + siginificance_folder_name + "/significance_indexes_[{},{},{},{},{}].py".format(best_threshold[0],best_threshold[1],best_threshold[2],best_threshold[3],best_threshold[4])
    significance_index_file_path = "./" + siginificance_folder_name + "/significance_indexes_[{},{},{}].py".format(best_threshold[0],best_threshold[1],best_threshold[2])
    modded_model_name = "modded_" + model_name +"_{},{},{}".format(best_threshold[0],best_threshold[1],best_threshold[2])
    # modded_model_name = "modded_" + model_name +"_{},{},{},{},{}".format(best_threshold[0],best_threshold[1],best_threshold[2],best_threshold[3],best_threshold[4])
    modded_folder_name = "modified_tlfite_models_{}_{}_{}".format(start/1000,(stop-1)/1000,step)
    modded_folder_path = "./" + modded_folder_name + "/"
    modded_model_path = modded_folder_path + modded_model_name + ".tflite"

    # write optimal results to a text file 
    f = open("./"+plot_folder+"/optimal_results_{}_{}_{}_{}.txt".format(start,stop,step,acc_threshold), "w")
    f.write(str(best_threshold)+"\n"+str(best_x)+"\n"+str(best_y)+"\n"+str(orig_acc))
    # annotation_text = f'Threshold: {best_threshold}'

    # copy the optimal model and correspodning significance indexes to the plot folder for easy access
    shutil.copy(modded_model_path, "./" + plot_folder + "/optimal_model_{}.tflite".format(acc_threshold))
    shutil.copy(significance_index_file_path, "./" + plot_folder + "/optimal_significance_{}.py".format(acc_threshold))
    # Highlight the corresponding point in green

    # Draw a line at minimum acc
    plt.figure(figsize=(WIDTH_SIZE,HEIGHT_SIZE))

    # Calculate the number of elements to keep (50% of the elements)
    num_elements_to_keep = len(x_values) // 10

    # Create an array of indices and shuffle them randomly
    indices = np.arange(len(x_values))
    np.random.shuffle(indices)

    # Select the first half of the shuffled indices to keep
    selected_indices = indices[:num_elements_to_keep]

    # Use the selected indices to create reduced arrays
    x_values_to_plot = x_values[selected_indices]
    y_values_to_plot = y_values[selected_indices]


    # Calculate the alpha values based on point density (overlap)
    _, _, density = np.histogram2d(x_values_to_plot, y_values_to_plot, bins=(100, 100))
    max_density = density.max()
    alpha_values = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * (density / max_density)

    # Create the diagram with scatter points
    plt.scatter(x_values_to_plot, y_values_to_plot, marker='o', s=MARKERSIZE,alpha=alpha_values)
    # print(pareto_optimal_solutions)
    plt.scatter(pareto_optimal_solutions[:, 0], pareto_optimal_solutions[:, 1], c='green', marker='^',s=MARKERSIZE, label='Pareto Optimal',alpha=1)
    # plt.title('Threshold Configuration Accuracy')
    plt.xlabel('Total Perforated MACs / Total MACs')
    plt.ylabel('Accuracy')
    # plt.scatter(best_x, best_y, color='green', marker='^', s=MARKERSIZE)
    plt.scatter(0, orig_acc, color='black', marker='x', s=MARKERSIZE,label='Original Accuracy')
    # plt.axhline(y=minimum_acc, color='r', linestyle='--', linewidth=0.5, label='Threshold Line')
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), fancybox=True, shadow=True, ncol=2)
    plt.grid(color='gray', linestyle='--', linewidth=0.1)
    # plt.grid(True)

    # plt.annotate(annotation_text, (best_x, best_y), textcoords="offset points", xytext=(-80,-200), ha='center',color='green')
    plt.savefig(plot_destination.format(start,stop,step,acc_threshold),bbox_inches="tight")

    plt.close()
