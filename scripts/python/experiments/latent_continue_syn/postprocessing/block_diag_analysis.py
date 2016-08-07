import numpy as np
import os
import sys
import glob
import re

def check_matrix(matrix):
    '''
    Only take squared matrix
    '''
    row, column = matrix.shape
    if (row != column):
        print ("Input Matrix is not a squared matrix!")
        sys.exit(-1)

def construct_symmetric_matrix(matrix):
    '''
    modify the original matrix to be a symmetric matrix that is suitable
    for RCM algorithm
    Way to do it is A + A.T with cirtical value 2*(1/#rows)
    '''
    row, column = matrix.shape
    critical_value = 1.0/row
    symmetric_matrix = matrix + matrix.T
    for i in range(row):
        for j in range(column):
            if (symmetric_matrix[i][j] <= critical_value * 2 and i != j):
                symmetric_matrix[i][j] = 0
            else:
                symmetric_matrix[i][j] = 1
    return symmetric_matrix
    

def find_lowest_degree_node(degree_node, set_of_node):
    '''
    to find the node of lowest degree in a given set
    '''
    left_node_degree = {node: degree_node[node] for node in set_of_node}
    lowest_degree = min(left_node_degree.values())
    for key in left_node_degree:
        if (left_node_degree[key] == lowest_degree):
            node_lowest_degree = key
            break
    return node_lowest_degree

def sort(degree_node, set_of_node):
    '''
    return nodes in a given set in an increasing order by degree
    '''
    input_node_degree = {node: degree_node[node] for node in set_of_node}
    sorted_degree = sorted(list(input_node_degree.values()))
    sorted_node = []
    index = 0
    while (len(sorted_node) != len(set_of_node)):
        for node in set_of_node:
            if (input_node_degree[node] == sorted_degree[index] and node not in sorted_node):
                sorted_node.append(node)
                index += 1
                break
    return sorted_node



def RCM(matrix):
    '''
    use Cuthill-McKee and RCM algorithms to find the correct permutation of
    row and column to resemble the block diagnoal structure
    return permutation R
    '''
    row, column = matrix.shape
    nodes = [i for i in range(row)]
    adjacent_node = {}
    for i in range(row):
        adjacent_node[i] = []
        for j in range(column):
            if (matrix[i][j] > 0 and i != j):
                adjacent_node[i].append(j)
    degree_node = {}
    for node in adjacent_node:
        degree_node[node] = len(adjacent_node[node])
    Q = []
    R = []
    while (len(R) != row):
        node_left = []
        for node in nodes:
            if (node not in R):
                node_left.append(node)
        P = find_lowest_degree_node(degree_node, node_left)
        R.append(P)
        for node in sort(degree_node, adjacent_node[P]):
            if (node not in R):
                Q.append(node)
        while (Q != []):
            C = Q[0]
            Q = Q[1:]
            if (C not in R):
                R.append(C)
                for node in sort(degree_node, adjacent_node[C]):
                    if (node not in R):
                        Q.append(node)
    R.reverse()
    return R


def get_block_diagonal(matrix, permutation):
    '''
    use the permutation output from RCM to resemble the block diagonal structure
    '''
    row, column = matrix.shape
    row_permuted = np.zeros((row, column))
    for i in range(len(permutation)):
        row_permuted[i,:] = matrix[permutation[i],:]
    block_diagonal = np.zeros((row, column))
    for i in range(len(permutation)):
        block_diagonal[:,i] = row_permuted[:,permutation[i]]
    return block_diagonal

def find_first_zero_entry(array):
    if (0 not in array):
        return len(array)
    else: 
        for i in range(len(array)):
            if (array[i] == 0):
                return i

def discover_grouping(matrix):
    '''
    discover the grouping from the block diagonal structure
    '''
    grouping = []
    while (matrix.shape != (0,0)):
        row, column = matrix.shape
        assumption = find_first_zero_entry(matrix[0])
        num_in_group = []
        num_in_group.append(assumption)
        for i in range(1,assumption):
            num_in_group.append(find_first_zero_entry(matrix[i]))
        NUM_in_GROUP = {}
        for i in num_in_group:
            if (NUM_in_GROUP.get(i, None) == None):
                NUM_in_GROUP[i] = 1
            else:
                NUM_in_GROUP[i] += 1
        max_number = max(NUM_in_GROUP.values())
        for key in NUM_in_GROUP:
            if (NUM_in_GROUP[key]==max_number):
                number = key
                break
        grouping.append(number)
        matrix = matrix[number:,number:]
    return grouping

def get_group(permutation, grouping):
    '''
    return which states are in which group
    '''
    group = {}
    for i in range(len(grouping)):
        group[i+1]=[]
        for j in range(grouping[i]):
            group[i+1].append(permutation[j])
        permutation = permutation[grouping[i]:]
    return group

def set_up_ouput_structure(results_root):
    output_root = os.path.join(results_root, 'G')
    try:
        os.mkdir(output_root)
    except OSError:
        print("G directory has been created!")
    output_block = os.path.join(output_root, 'block_A')
    try:
        os.mkdir(output_block)
    except OSError:
        print("Block_A directory has been created under G!")
    output_one_zero = os.path.join(output_root, 'zero_one')
    try:
        os.mkdir(output_one_zero)
    except OSError:
        print("zero_one directory has been created under G!")
    output_grouping = os.path.join(output_root, 'grouping')
    try:
        os.mkdir(output_grouping)
    except OSError:
        print("grouping directory has been created under G!")

def read_matrix(path):
    iteration_num = re.search(r'\d+', path.split('/')[-1]).group()
    try:
        matrix = np.loadtxt(path)
        print(iteration_num, " matrix imported!")
    except:
        print(iteration_num, " matrix not found!")
        sys.exit(-1)
    return matrix, iteration_num

def output(block_matrix, block_diagonal, group, number, results_root):
    output_block = os.path.join(results_root, 'G/block_A')
    output_one_zero = os.path.join(results_root, 'G/zero_one')
    output_grouping = os.path.join(results_root, 'G/grouping')
    block_matrix_file = open(os.path.join(output_block, number+'.txt'), 'w')
    row, column = block_matrix.shape
    for i in range(row):
        for j in range(column):
            block_matrix_file.write(str(block_matrix[i,j]))
            block_matrix_file.write(" ")
        block_matrix_file.write("\n")
    block_matrix_file.close()
    block_diagonal_file = open(os.path.join(output_one_zero, number+'.txt'), 'w')
    for i in range(row):
        for j in range(column):
            block_diagonal_file.write('%d ' %(block_diagonal[i,j]))
        block_diagonal_file.write('\n')
    block_diagonal_file.close()
    grouping_file = open(os.path.join(output_grouping, number+'.txt'), 'w')
    for key in group:
        for st in group[key]:
            grouping_file.write('%d %d\n' %(st, key))
    grouping_file.close()
    print('All Files outputed!')
            
                



#main
experiment = str(input("Experiment Name: "))

results_root = os.path.join('results',experiment)

matrix_root = os.path.join(results_root, 'A')

set_up_ouput_structure(results_root)

for file_path in glob.glob(os.path.join(matrix_root, '*.txt')):
    matrix, number = read_matrix(file_path)
    check_matrix(matrix)
    symmetric_matrix = construct_symmetric_matrix(matrix)
    permutation = RCM(symmetric_matrix)
    block_diagonal = get_block_diagonal(symmetric_matrix, permutation)
    block_matrix = get_block_diagonal(matrix, permutation)
    grouping = discover_grouping(block_diagonal)
    group = get_group(permutation, grouping)
    output(block_matrix, block_diagonal, group, number, results_root)






