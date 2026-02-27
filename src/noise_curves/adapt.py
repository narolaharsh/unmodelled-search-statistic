import numpy as np
import matplotlib.pyplot as plt


"""
Script to adapt the ET_D_psd.txt (which is meant for 10km triangle) for a 15km L. 

The formula I use is the following 

output_psd = input_psd * np.power(input_arm_length/output_arm_length, 2)
"""




def main():
    _input_psd = "ET_D_psd.txt"

    input_arm_length = 10.0
    output_arm_length = 15.0

    scaling_factor = (input_arm_length / output_arm_length)**2
    
    input_psd = np.genfromtxt(_input_psd, unpack=True)

    output_psd = input_psd[1] * scaling_factor

    np.savetxt("./ET_D_psd_15km.txt", np.column_stack([input_psd[0], output_psd]))

    return None


if __name__ == "__main__":
    main()
