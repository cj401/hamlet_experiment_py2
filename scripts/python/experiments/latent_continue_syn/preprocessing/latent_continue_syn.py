import numpy as np
import random
import configparser
import os
import sys

__autor__='bill'

def probability(num, constant):
    prob = []
    for i in range(num):
        prob.append(random.gammavariate(1*constant, 1/constant/num))
    sum_prob = sum(prob)
    for i in range(num):
        prob[i] /= sum_prob
    return prob

def gen_state(prob):
    u = random.uniform(0,1)
    a = 0
    for i in range(len(prob)):
        a += prob[i]
        if (u <= a):
            return i

def output_file(which, data, path):
    file = open(path, 'w')
    if (which == 'matrix'):
        for i in range(len(data)):
            for j in range(len(data[i])):
                file.write(str(data[i][j]))
                file.write(" ")
            file.write("\n")
    elif (which == 'array'):
        for i in range(len(data)):
            file.write(str(data[i]))
            file.write("\n")
    elif (which == 'dict'):
        for key in data:
            file.write(str(key))
            file.write(" ")
            file.write(str(data[key]))
            file.write("\n")
    file.close()


class Data(object):
    def __init__(self, config_file_path):
        config = configparser.ConfigParser()
        config.read(config_file_path)
        self.groupnumber = int(config.get('parameter', 'number_of_group'))
        self.num_state_in_group = int(config.get('parameter','number_state_in_group'))
        self.num_state = self.groupnumber * self.num_state_in_group
        self.num_state_out_group = self.num_state - self.num_state_in_group
        self.prob_in_group = float(config.get('parameter','trans_prob_in_group'))
        self.prob_out_group = 1 - self.prob_in_group
        self.time_step = int(config.get('parameter','time_step'))
        self.dimension = int(config.get('parameter','data_dimension'))
        self.mean_sd = int(config.get('parameter','mean_sd'))
        self.gamma_const = float(config.get('parameter','gamma_constant'))
        self.save_path = str(config.get('output','save_path'))

    def transition_matrix(self):
        self.group_assignment = {}
        for i in range(self.groupnumber):
            for j in range(self.num_state_in_group * i, self.num_state_in_group * (i+1)):
                self.group_assignment[j]=i
        self.trans = np.zeros((self.num_state, self.num_state))
        for i in range(self.num_state):
            prob_inside_group = probability(self.num_state_in_group, self.gamma_const)
            prob_outside_group = probability(self.num_state_out_group, self.gamma_const)
            in_group = 0
            out_group = 0
            for j in range(self.num_state):
                if (self.group_assignment[i] == self.group_assignment[j]):
                    self.trans[i][j] = self.prob_in_group * prob_inside_group[in_group]
                    in_group += 1
                else:
                    self.trans[i][j] = self.prob_out_group * prob_outside_group[out_group]
                    out_group += 1

    def gen_precision(self):
        precision_diag = []
        cov_diag = []
        for i in range(self.dimension):
            precision_entry = random.gammavariate(1,1)
            precision_diag.append(precision_entry)
            cov_diag.append(pow(1/precision_entry, 0.5))
        self.precision = np.diag(precision_diag)
        self.cov = np.diag(cov_diag)

    def gen_mean(self):
        self.mean = []
        for i in range(self.num_state):
            mean_state = []
            for j in range(self.dimension):
                mean_state.append(random.gauss(0,self.mean_sd))
            self.mean.append(mean_state)

    def gen_state(self):
        self.test_state = []
        self.train_state = []
        self.test_state.append(random.randint(0, self.num_state - 1))
        self.train_state.append(random.randint(0, self.num_state - 1))
        self.test_mean_by_time_step = []
        self.train_mean_by_time_step = []
        self.test_obs = []
        self.train_obs = []
        for i in range(self.time_step):
            test_previous = self.test_state[-1]
            train_previous = self.train_state[-1]
            new_test_state = gen_state(self.trans[test_previous])
            new_train_state = gen_state(self.trans[train_previous])
            self.test_state.append(new_test_state)
            self.train_state.append(new_train_state)
            self.test_mean_by_time_step.append(self.mean[new_test_state])
            self.train_mean_by_time_step.append(self.mean[new_train_state])
            test_data = np.random.multivariate_normal(np.array(self.test_mean_by_time_step[-1]).T, self.cov)
            self.test_obs.append(test_data)
            train_data = np.random.multivariate_normal(np.array(self.train_mean_by_time_step[-1]).T, self.cov)
            self.train_obs.append(train_data)
        self.test_state.remove(self.test_state[0])
        self.train_state.remove(self.train_state[0])

    def output(self, output_file_name):
        try:
            os.mkdir(os.path.join(self.save_path, output_file_name))
        except OSError:
            print(output_file_name, " exist in ", self.save_path)
        self.save_path = os.path.join(self.save_path, output_file_name)
        output_file('matrix', self.trans, os.path.join(self.save_path, 'A.txt'))
        output_file('matrix', self.precision, os.path.join(self.save_path, 'noise_sd.txt'))
        output_file('matrix', self.mean, os.path.join(self.save_path, 'mean.txt'))
        output_file('array', self.test_state, os.path.join(self.save_path, 'test_z.txt'))
        output_file('array', self.train_state, os.path.join(self.save_path, 'z.txt'))
        output_file('matrix', self.test_obs, os.path.join(self.save_path, 'test_obs.txt'))
        output_file('matrix', self.train_obs, os.path.join(self.save_path, 'obs.txt'))
        output_file('dict', self.group_assignment, os.path.join(self.save_path, 'group.txt'))
        try:
            os.mkdir(os.path.join(self.save_path, 'S'))
        except OSError:
            print("S exist in ", self.save_path)
        S_path = os.path.join(self.save_path, 'S')
        output_file('matrix', self.test_mean_by_time_step, os.path.join(S_path, 'test.txt'))
        output_file('matrix', self.train_mean_by_time_step, os.path.join(S_path, 'train.txt'))



def main(argv):
    directory = '/'.join(str(argv[0]).split('/')[:-1])
    config_file_path = os.path.join(directory, str(argv[1]))
    output_file_name = str(argv[2])
    new_data = Data(config_file_path)
    new_data.transition_matrix()
    new_data.gen_precision()
    new_data.gen_mean()
    new_data.gen_state()
    new_data.output(output_file_name)

if __name__=="__main__":
    main(sys.argv)

        


