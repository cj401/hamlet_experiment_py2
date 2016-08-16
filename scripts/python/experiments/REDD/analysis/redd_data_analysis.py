# Analysis of performance on REDD data after fitting a hamlet model
# Use evaluation metric from jw2013


import numpy as np
import sys
import itertools
import random
import os
import glob
import re

__author__='bill'

"""
README:
For the program to run correctly (in terminal):
() python *file.py results_directory groundtruth_path
() resutls_directory will be the ... part in results/.../mean_by_state
() groundtruth_path will be ... in data/...(all the way to the txt file)
() need to be run in the same level with results or data,
   e.g. in hamlet/ or hamlet/experiment directory
"""

"""
Find best pairing between two sets of positive real vectors
"Best" means by maximizing the accuracy value from the error metric of Kolter and
Johnson (2011) given in Bayesian Nonparametric HSMM (Johnson and Willsky, 2013).

The core script does the following:
() Reads 'amplitudes.txt' from /data/data/.../ (ground-truth matrix)
() Iterates through each '*.txt' file in /results/.../mean_by_state/ (predictive matrix)
    ()check whether predictive matrix and groundtruth matrix have the same shape
    ()calculate the difference between columns in predictive and columns in ground-truth,
      for all time interval, which is the sum_{t=1}^{T}|y_{predictive}-y_{ground-truth}|.
      Store the ouput in a dictionary called 'diff_dict'
    ()compute the numerator in the Kolter and Johnson by going through all
      possible permutations and store the smallest sum and its permutation
    ()compute the accuarcy value
() Create 'test_Kolter_and_Johnson_accuarcy' and 'train_Kolter_and_Johnson_accuarcy'
   under /results/.../ to store the accuarcy value of each iteration
() Create 'test_Kolter_and_Johnson_permutation' and 'train_Kolter_and_Johnson_permutation'
   under /results/.../ to store the permutation that yields the maximum accuracy value
   in each iteration
() Format of the permutation is the following:
   For a 5 columns matrix comparison, suppose the output is '5 1 3 2 4', that is
   predictive -> ground-truth
   column 1   -> column 5
   column 2   -> column 1
   column 3   -> column 3
   column 4   -> column 2
   column 5   -> column 4
() If groundtruth contain negative file, it will output a .log file specifying that.
"""

#-------------------------------------------------------------------------------
#Read Ground-truth matrix and Predictive matrix

def read_groundtruth(path):
    """
    input the path to the directory for the ground-truth matrixs, e.g. /data/.../mean_by_state/
    return the test_ground_truth and train_ground_truth
    """
    negative = False
    try:
        groundtruth = np.loadtxt(path)
        print("Test_Ground_Truth imported!")
    except OSError:
        print("Cannot find groundtruth file ", path)
        sys.exit(-1)
    if (any(i < 0 for i in groundtruth.flatten())):
        print("Test_Ground_Truth contian negative value!")
        negative = True
    return groundtruth, negative


def read_predictive(path):
    """
    input the full path to the predictive file e.g /data/.../mean_by_state/*.txt
    return the predictive matrix
    """
    iteration_num = re.search(r'\d+', path.split('/')[-1]).group()
    try:
        predictive = np.loadtxt(path)
        print(iteration_num, " predictive matrix imported!")
    except:
        print(iteration_num, " predictive matrix cannot be imported!")
        sys.exit(-1)
    return predictive, iteration_num

#-------------------------------------------------------------------------------
#functions for calculating the diff_dict

def is_same_shape(matrix1, matrix2):
    if matrix1.shape == matrix2.shape:
        return True
    return False


def get_diff_dict(predictive, ground_truth):
    """
    Input the predictive and the Ground-truth matrix and calculate the absolute
    difference between columns in predictive and each column in the ground_truth.
    Suppose there are n columns, it is n*n computation.
    Return a diff_dict in the following format:
    {column1(predictive):{column1(ground_truth):absolute difference for all t,
    column2(ground_truth): absolute difference for all t ...}...}
    """

    row, column = predictive.shape
    diff_dict={}
    for i in range(1, column+1):
        diff_dict[i] = {}
        for j in range(1, column+1):
            diff_dict[i][j] = sum(abs(predictive[:,i-1]-ground_truth[:,j-1]))
    print("diff_dict ready!")
    return diff_dict


#-------------------------------------------------------------------------------
#function for calculating the numerator part in the Kolter and Johnson by running
#through all possible permutations of the columns

def get_numerator(diff_dict):
    """
    Input the 'diff_dict' dictionary and the number of columns
    Running through all possible permutations and return the smallest value and
    the associated permutation
    """
    permutation_object = list(diff_dict.keys())
    permutation = list(itertools.permutations(permutation_object,len(permutation_object)))
    print("Start Permutation!")
    sum_permutation_diff={}
    for i in range(len(permutation)):
        difference = 0
        k = 0
        for j in permutation[i]:
            k += 1
            difference += diff_dict[k][j]
        if (i == 0):
            smallest_numerator = difference
            smallest_permutation = permutation[i]
        else:
            if (difference < smallest_numerator):
                smallest_numerator = difference
                smallest_permutation = permutation[i]
        sum_permutation_diff[i]={'difference':difference, 'permutation':permutation[i]}
        #sum_permutation_diff exist for double checking, but not necessary
    print("Permutation Done!")
    return smallest_numerator, smallest_permutation

#-------------------------------------------------------------------------------
#calculating the accuracy value

def get_denominator(goundtruth):
    return 2.0 * sum(sum(abs(goundtruth)))

def get_accuracy_value(numerator, denominator):
    return 1 - float(numerator / denominator)

#-------------------------------------------------------------------------------
#iterate through all *.txt in /results/.../mean_by_state

def iterate_all_predictive(path, groundtruth):
    output = {}
    for file_path in glob.glob(os.path.join(path,'*.txt')):
        predictive, iteration_num = read_predictive(file_path)
        if (is_same_shape(predictive, groundtruth) == True):
            numerator, permutation = get_numerator(get_diff_dict(predictive, groundtruth))
            accuracy_value = get_accuracy_value(numerator, get_denominator(groundtruth))
        else:
            print("Predictive and Ground-truth are not the same shape!")
            sys.exit(-1)
        output[iteration_num] = {"accuracy":accuracy_value, "permutation":permutation}
    return output

#-------------------------------------------------------------------------------
#output accuracy value and associated permutation to /results/.../

def output_file(output, path, name, negative):
    accuracy_file_name = str(name + '_Kolter_and_Johnson_accuracy.txt')
    permutation_file_name = str(name + '_Kolter_and_Johnson_permutation.txt')
    try:
        accuracy_file = open(os.path.join(path, accuracy_file_name), 'w')
    except OSError:
        print("Output Path Incorrect!")
        sys.exit(-1)
    try:
        permutation_file = open(os.path.join(path, permutation_file_name), 'w')
    except OSError:
        print("Output Path Incorrect!")
        sys.exit(-1)
    for iteration in output:
        accuracy_file.write(iteration)
        accuracy_file.write("      ")
        accuracy_file.write("%.4f \n" %(output[iteration]['accuracy']))
        permutation_file.write(iteration)
        permutation_file.write("      ")
        permutation_str = ' '.join(map(str, output[iteration]['permutation']))
        permutation_file.write(permutation_str)
        permutation_file.write("\n")
    accuracy_file.close()
    permutation_file.close()
    if (negative == True):
        try:
            negative_log = open(os.path.join(path, name+'.log'), 'w')
        except OSError:
            print("Output Path Incorrect!")
            sys.exit(-1)
        negative_log.write("GroundTruth matrix contains negative value!")
        negative_log.close()
    print("Output ", accuracy_file_name," ", permutation_file_name, " and ", name, ".log file!")
        

def main(argv):
    name = str(argv[1].split('/')[-1].split('.')[0])
    groundtruth_path = os.path.join('data', argv[1])
    output_path = os.path.join('results', argv[0])
    predictive_directory_path = os.path.join(output_path, 'mean_by_state')
    groundtruth, negative = read_groundtruth(groundtruth_path)
    output_test = iterate_all_predictive(predictive_directory_path, groundtruth)
    output_file(output_test, output_path, name, negative)
    
    

if __name__=="__main__":
    main(sys.argv[1:])


    


        


    
 

