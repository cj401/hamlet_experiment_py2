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
() Run the program in /hamlet/experiment directory
() Type in the directory name for the dataset
   e.g the ... part in results/.../ or data/.../
() Then the program will run automatically and output file in /results/.../
"""


"""
Find best pairing between two sets of positive real vectors
"Best" means by maximizing the accuracy value from the error metric of Kolter and
Johnson (2011) given in Bayesian Nonparametric HSMM (Johnson and Willsky, 2013).

The core script does the following:
() Reads 'test.txt' and 'train.txt' from /data/.../mean_by_state/ (ground-truth matrix)
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
"""

#-------------------------------------------------------------------------------
#Read Ground-truth matrix and Predictive matrix

def read_ground_truth(path):
    """
    input the path to the directory for the ground-truth matrixs, e.g. /data/.../mean_by_state/
    return the test_ground_truth and train_ground_truth
    """
    try:
        test_ground_truth = np.loadtxt(os.path.join(path,'test.txt'))
        print("Test_Ground_Truth imported!")
    except OSError:
        print("No test.txt file in ", path)
        sys.exit(-1)
    if (any(i < 0 for i in test_ground_truth.flatten())):
        test_log = 1
    else:
        test_log = 0
    try:
        train_ground_truth = np.loadtxt(os.path.join(path,'train.txt'))
        print("Train_Ground_Truth imported!")
    except OSError:
        print("No train.txt file in ", path)
        sys.exit(-1)
    if (any(i < 0 for i in train_ground_truth.flatten())):
        train_log = 1
    else:
        train_log = 0
    return test_ground_truth, train_ground_truth, test_log, train_log


def read_predictive(path):
    """
    input the full path to the predictive file e.g /data/.../mean_by_state/*.txt
    return the predictive matrix
    """
    iteration_num = re.search(r'\d+', path).group()
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

def output_file(output, path, log, which):
    accuracy_file_name = str(which + '_Kolter_and_Johnson_accuracy.txt')
    permutation_file_name = str(which + '_Kolter_and_Johnson_permutation.txt')
    log_file_name = str(which + '_log.txt')
    try:
        accuracy_file = open(os.path.join(path, accuracy_file_name), 'w')
    except OSError:
        print("Output Path Incorrect!")
    try:
        permutation_file = open(os.path.join(path, permutation_file_name), 'w')
    except OSError:
        print("Output Path Incorrect!")
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
    try:
        log_file = open(os.path.join(path, log_file_name),'w')
    except OSError:
        print("Output Path Incorrect!")
    log_file.write(which)
    if (log == 1):
        log_file.write(" ground truth matrix contains some negative entries.\n")
    elif (log == 0):
        log_file.write(" ground truth matrix contains only positive entries.\n")
    log_file.close()      
    print("Output: \n", accuracy_file_name, ",\n", permutation_file_name, ",\n", log_file_name, ".\n")


#------------------------------------------------------------------------------
#Main
experiment_name = str(input("Dataset name (e.g. the ... for '/results/...'): "))
ground_truth_path = os.path.join(os.path.join('data',experiment_name),'mean_by_state')
output_path = os.path.join('results',experiment_name)
predictive_directory_path = os.path.join(output_path, 'mean_by_state')
test_ground_truth, train_ground_truth, test_log, train_log = read_ground_truth(ground_truth_path)
#test_ground_truth analysis
output_test = iterate_all_predictive(predictive_directory_path, test_ground_truth)
output_file(output_test, output_path, test_log, 'test')
#train_ground_truth analysis
output_train = iterate_all_predictive(predictive_directory_path, train_ground_truth)
output_file(output_train, output_path, train_log, 'train')
    
    
