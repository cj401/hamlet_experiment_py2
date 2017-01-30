import sys
import os
import numpy as np
import glob
import re

__author__='bill'
'''
README:
For the program to run accurately (in terminal):
() Run the program in /hamlet directory
() The command is
   python *file.py directory_name_for_A true/false(for reduce) m= t=
   where m is the number of occurence the states occur in n_dot (default is 0),
   t is the threshold multiple user specified (default is 2)
() e.g. python *path/block_diag_analysis.py continuous_latent/syn_10000itr_hmc_diag40/
        block_diag40_s2/LT_hdp_hmm_w0/01 true
        this will block diagonize matrix in that directory with reduction and
        default parameter value m=0, t=2
() To change default value, simply add m=value1 t=value2 after '... true'
   in command line 
'''


def check_matrix(matrix):
    '''
    Only take squared matrix
    '''
    row, column = matrix.shape
    if (row != column):
        print ("Input Matrix is not a squared matrix!")
        sys.exit(-1)

def reduce_matrix(matrix, keep):
    '''
    Delete the rows and columns that are not wanted
    '''
    row_reduce = matrix[keep,:]
    full_reduce = row_reduce[:,keep]
    return full_reduce

def normalized_matrix(matrix):
    '''
    normalize the rows of the input matrix
    '''
    row, column = matrix.shape
    for i in range(row):
        row_sum = sum(matrix[i,:])
        for j in range(column):
            matrix[i,j] /= row_sum
    return matrix
        
def construct_symmetric_matrix(matrix, tm = 2):
    '''
    modify the original matrix to be a symmetric matrix that is suitable
    for RCM algorithm
    Way to do it is A + A.T with cirtical value tm(for threshold multiple)*(1/#rows)
    '''
    row, column = matrix.shape
    critical_value = 1.0/float(row)
    symmetric_matrix = matrix + matrix.T
    for i in range(row):
        for j in range(column):
            if (symmetric_matrix[i][j] <= tm * critical_value and i != j):
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

def read_matrix(path):
    '''
    Read transition matrix.
    '''
    iteration_num = re.search(r'\d+', path.split('/')[-1]).group()
    try:
        matrix = np.loadtxt(path)
        print(iteration_num, " matrix imported!")
    except:
        print(iteration_num, " matrix not found!")
        sys.exit(-1)
    return matrix, iteration_num

def get_n_dot(path, min_occurence = 0):
    '''
    get_n_dot file, and user specify the min_occurence.
    If state occurence greater than min_occurence, keep state.
    Else, delete the state
    '''
    n_dot_file = open(os.path.join(path, 'n_dot.txt'), 'r')
    output_keep_file = open(os.path.join(path, 'G/keep.txt'), 'w')
    output_keep_file.write('iteration states_keep\n')
    n_dot = n_dot_file.readlines()[1:]
    n_dot_file.close()
    keep = {}
    for line in n_dot:
        iteration = line.split()[0]
        output_keep_file.write(iteration)
        output_keep_file.write("    ")
        to_keep = []
        i = 0
        for value in line.split()[1:]:
            if (int(value) > min_occurence):
                output_keep_file.write(" %d" %(i))
                to_keep.append(i)
            i += 1
        output_keep_file.write("\n")
        keep[iteration] = to_keep
    return keep

def set_up_output_structure(results_root):
    '''
    set up the output directory structure
    '''
    output_root = os.path.join(results_root, 'G')
    try:
        os.mkdir(output_root)
    except OSError:
        if (os.path.exists(results_root)==False):
            print(results_root, " does not exist.")
            sys.exit(-1)
        else:
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
    output_permutation = os.path.join(output_root, 'permutation')
    try:
        os.mkdir(output_permutation)
    except OSError:
        print("Permutation directory has been created under G!")
        
    '''
    Remove grouping directory since the grouping algorithm might not
    be accuarate given the heatmap of the block-diagonalized matrix
    I will think about a better algorithm, but before that, I will remove
    that part from the code
    '''

def output(block_matrix, block_diagonal, permutation, number, results_root):
    '''
    output files
    '''
    output_block = os.path.join(results_root, 'G/block_A')
    output_one_zero = os.path.join(results_root, 'G/zero_one')
    output_permutation = os.path.join(results_root, 'G/permutation')
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
    permutation_file = open(os.path.join(output_permutation, number+'.txt'), 'w')
    for i in permutation:
        permutation_file.write('%d\n' %(i))
    permutation_file.close()
    print('All Files related to iteration ', number, 'outputed!')
    
def main(argv):
    experiment = str(argv[0])
    reduce = str(argv[1]).lower()
    for arg in argv[2:]:
        if ('m' in arg):
            m = int(arg.split('=')[1])
        elif('t' in arg):
            t = float(arg.split('=')[1])
    #results_root = os.path.join('results',experiment)
    results_root = experiment
    matrix_root = os.path.join(results_root, 'A')
    set_up_output_structure(results_root)
    if (reduce == 'true'):
        try:
            keep_row_column = get_n_dot(results_root, m)
        except:
            keep_row_column = get_n_dot(results_root)
    for file_path in glob.glob(os.path.join(matrix_root,'*.txt')):
        matrix, number = read_matrix(file_path)
        if (reduce == 'true'):
            matrix = normalized_matrix(reduce_matrix(matrix, keep_row_column[str(number)]))
        check_matrix(matrix)
        try:
            symmetric_matrix = construct_symmetric_matrix(matrix, t)
        except:
            symmetric_matrix = construct_symmetric_matrix(matrix)
        permutation = RCM(symmetric_matrix)
        block_diagonal = get_block_diagonal(symmetric_matrix, permutation)
        block_matrix = get_block_diagonal(matrix, permutation)
        output(block_matrix, block_diagonal, permutation, number, results_root)
    

if __name__=="__main__":
    main(sys.argv[1:])



