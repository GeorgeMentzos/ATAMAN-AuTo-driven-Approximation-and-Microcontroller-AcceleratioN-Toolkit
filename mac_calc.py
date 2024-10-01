mat_mult_counter = [450,84,8]
total_macs = [1728,36864,73728]

starting_macs = 4464000

mac_reduction_dsp = [[0,10960,26948],[0,13140,33572],[0,15148,37804],[0,21216,37804]]

for array in mac_reduction_dsp:
    final_macs = starting_macs

    for i in range(len(mat_mult_counter)):
        final_macs -= mat_mult_counter[i]*array[i]

    print(final_macs)
    print("------")