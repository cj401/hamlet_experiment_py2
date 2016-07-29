import os
import datetime
import collections
import glob
import numpy as np

__author__ = 'clayton'


# ----------------------------------------------------------------------

def hello():
    print 'Hello from utilities.util.hello()!'


# ----------------------------------------------------------------------

# From http://stackoverflow.com/questions/11351032/named-tuple-and-optional-keyword-arguments
def namedtuple_with_defaults(typename, field_names, default_values=None):
    if default_values is None:
        default_values = []
    T = collections.namedtuple(typename, field_names)
    T.__new__.__defaults__ = (None,) * len(T._fields)
    if isinstance(default_values, collections.Mapping):
        prototype = T(**default_values)
    else:
        prototype = T(*default_values)
    T.__new__.__defaults__ = tuple(prototype)
    return T


# ----------------------------------------------------------------------

def get_timestamp(verbose=False):
    now = datetime.datetime.now()
    # '{0}{month:02d}{2}_{3}{4}{5}'.format(now.year, month=now.month, now.day, now.hour, now.minute, now.second)
    if verbose:
        return 'y={year:04d},m={month:02d},d={day:02d}_h={hour:02d},m={minute:02d},s={second:02d},mu={micro:06d}' \
            .format(year=now.year, month=now.month, day=now.day,
                    hour=now.hour, minute=now.minute, second=now.second,
                    micro=now.microsecond)
    else:
        return '{year:04d}{month:02d}{day:02d}_{hour:02d}{minute:02d}{second:02d}{micro:06d}' \
            .format(year=now.year, month=now.month, day=now.day,
                    hour=now.hour, minute=now.minute, second=now.second,
                    micro=now.microsecond)


# ----------------------------------------------------------------------

def flatten1(l):
    """
    Flattens list one level.
    E.g.,
    >>> flatten([ [1,2,3], [4,5,6], [7,8,9]])
    [1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> flatten([ [1,2,3], [4, ['a','b','c'], 5,6], [7,8,9]])
    [1, 2, 3, 4, ['a', 'b', 'c'], 5, 6, 7, 8, 9]
    :param l: list of lists
    :return: list flattened by one level
    """
    return [item for sublist in l for item in sublist]


# ----------------------------------------------------------------------

def center_scale_data(data, mean=0, var=1.0):
    """
    Centers the figures around mean and scales with variance var
    :param data:
    :param mean: figures center
    :param var: scale variance
    :return: centered and scaled figures
    """
    data_mean = np.mean(data) + mean
    data_variance = np.var(data) * var
    return [ ( ( datum - data_mean ) / data_variance ) for datum in data ]


def center_scale_data_lists(data_list, mean=0, std=1.0):
    """
    Given a list of figures, finds the mean and stdev of all of the figures across lists, then
    centers the figures around mean and scales with stdev var
    :param data_list: list of figures
    :param mean: figures center
    :param std: scale standard deviation
    :return: centered and scaled figures
    """
    all_data = flatten1(data_list)
    data_mean = np.mean(all_data) + mean
    data_std = np.std(all_data) * std

    # print 'data_mean:', data_mean
    # print ' data_std:', data_std

    return np.array([ [ (datum - data_mean) / data_std
                        for datum in data ]
                      for data in data_list ])


def center_scale_data_array(data_array, mean=0, std=1.0):
    """
    Same as above, but for numpy array
    :param data_array:
    :param mean:
    :param std:
    :return:
    """
    data_mean = np.mean(data_array) + mean
    data_std = np.std(data_array) * std
    return ( data_array - data_mean ) / data_std, data_mean, data_std


# ----------------------------------------------------------------------

def read_directory_files(directory_path):
    if directory_path[-1] != '/':
        directory_path += '/'
    files = list()
    for filepath in glob.glob(directory_path + '*'):
        files.append(filepath.split('/')[-1])
    return sorted(files)


# ----------------------------------------------------------------------

def is_results_dir(subdirList, fileList):
    # print 'subdirList', subdirList
    # print 'fileList', fileList
    for dirname in ['thetastar']:
        if dirname not in subdirList:
            return False
    for filename in ['F1_score.txt', 'accuracy.txt', 'precision.txt', 'recall.txt',
                     'parameters.config']:
        if filename not in fileList:
            return False
    return True


def is_data_dir(subdirList, fileList):
    for dirname in []:
        if dirname not in subdirList:
            return False
    for filename in ['obs.txt', 'states.txt', 'weights.txt']:
        if filename not in fileList:
            return False
    return True


def get_directory_branches(root_dir, dir_type_test_p=is_results_dir,
                           remove_root_p=False, main_path='../'):

    if main_path:
        owd = os.getcwd()
        os.chdir(main_path)

    dir_branches = list()

    for dirName, subdirList, fileList in os.walk(root_dir):
        if dir_type_test_p(subdirList, fileList):
            if remove_root_p:
                # print 'dirName', dirName, 'subdirList', subdirList, 'fileList', fileList
                without_root = dirName.replace(root_dir, '')  # + '/'
                # print 'without_root', without_root
                dir_branches.append(without_root)
            else:
                dir_branches.append(dirName)  #  + '/')

    if main_path:
        os.chdir(owd)

    return dir_branches


# ----------------------------------------------------------------------

def read_parameter_file_as_dict(parameter_filepath, parameter_filename, main_path=None):

    if parameter_filepath[-1] != '/':
        parameter_filepath += '/'

    config_filepath = parameter_filepath + parameter_filename

    '''
    glob_search_path = parameter_filepath + '/*.config'
    config_filepath = glob.glob(glob_search_path)
    if not config_filepath:
        print 'Could not find config file in this glob search path:\n' \
            + '    \'{0}\''.format(glob_search_path)
        sys.exit(1)
    '''

    params_dict = dict()

    if main_path:
        owd = os.getcwd()
        os.chdir(main_path)

    try:
        with open(config_filepath, 'r') as cfile:
            for line in cfile.readlines():
                line = line.rstrip('\n').split(' ')
                if line[0] and not ( line[0][0] == '/' ):
                    # print '>>> {0}'.format(line)
                    key = line[0] + ':' + line[1]
                    val = line[2]
                    params_dict[key] = val
                # else : print '<<< {0}'.format(line)
    except IOError, err:
        print 'ERROR: read_parameter_file_as_dict()'
        print '     Current working directory:', os.getcwd()
        raise IOError(err)

    if main_path:
        os.chdir(owd)

    return params_dict


# ----------------------------------------------------------------------

def pprint_dict(d):
    for key, val in d.iteritems():
        print '\'{0}\':\'{1}\''.format(key, val)


# ----------------------------------------------------------------------

def read_and_eval_file(filename):
    """
    SECURITY WARNING: don't eval any file you don't know/trust

    Intended use: read dictionary saved as text in file

    :param filename:
    :return:
    """
    contents = open(filename, 'r').read()
    return eval(contents)


# ----------------------------------------------------------------------

def sign(x):
    """
    Returns 1 or -1 depending on the sign of x
    """
    if (x >= 0):
        return 1
    else:
        return -1


# ----------------------------------------------------------------------

class Counter(dict):
    """
    A counter keeps track of counts for a set of keys.

    The counter class is an extension of the standard python
    dictionary type.  It is specialized to have number values
    (integers or floats), and includes a handful of additional
    functions to ease the task of counting figures.  In particular,
    all keys are defaulted to have value 0.  Using a dictionary:

    a = {}
    print a['test']

    would give an error, while the Counter class analogue:

    >>> a = Counter()
    >>> print a['test']
    0

    returns the default 0 value. Note that to reference a key
    that you know is contained in the counter,
    you can still use the dictionary syntax:

    >>> a = Counter()
    >>> a['test'] = 2
    >>> print a['test']
    2

    This is very useful for counting things without initializing their counts,
    see for example:

    >>> a['blah'] += 1
    >>> print a['blah']
    1

    The counter also includes additional functionality useful in implementing
    the classifiers for this assignment.  Two counters can be added,
    subtracted or multiplied together.  See below for details.  They can
    also be normalized and their total count and arg max can be extracted.
    """

    def __getitem__(self, idx):
        self.setdefault(idx, 0)
        return dict.__getitem__(self, idx)

    def incrementAll(self, keys, count):
        """
        Increments all elements of keys by the same count.

        >>> a = Counter()
        >>> a.incrementAll(['one','two', 'three'], 1)
        >>> a['one']
        1
        >>> a['two']
        1
        """
        for key in keys:
            self[key] += count

    def argMax(self):
        """
        Returns the key with the highest value.
        """
        if len(self.keys()) == 0: return None
        all = self.items()
        values = [x[1] for x in all]
        maxIndex = values.index(max(values))
        return all[maxIndex][0]

    def sortedKeys(self):
        """
        Returns a list of keys sorted by their values.  Keys
        with the highest values will appear first.

        >>> a = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> a['third'] = 1
        >>> a.sortedKeys()
        ['second', 'third', 'first']
        """
        sortedItems = self.items()
        compare = lambda x, y: sign(y[1] - x[1])
        sortedItems.sort(cmp=compare)
        return [x[0] for x in sortedItems]

    def totalCount(self):
        """
        Returns the sum of counts for all keys.
        """
        return sum(self.values())

    def normalize(self):
        """
        Edits the counter such that the total count of all
        keys sums to 1.  The ratio of counts for all keys
        will remain the same. Note that normalizing an empty
        Counter will result in an error.
        """
        total = float(self.totalCount())
        if total == 0: return
        for key in self.keys():
            self[key] = self[key] / total

    def divideAll(self, divisor):
        """
        Divides all counts by divisor
        """
        divisor = float(divisor)
        for key in self:
            self[key] /= divisor

    def copy(self):
        """
        Returns a copy of the counter
        """
        return Counter(dict.copy(self))

    def __mul__(self, y):
        """
        Multiplying two counters gives the dot product of their vectors where
        each unique label is a vector element.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['second'] = 5
        >>> a['third'] = 1.5
        >>> a['fourth'] = 2.5
        >>> a * b
        14
        """
        sum = 0
        x = self
        if len(x) > len(y):
            x, y = y, x
        for key in x:
            if key not in y:
                continue
            sum += x[key] * y[key]
        return sum

    def __radd__(self, y):
        """
        Adding another counter to a counter increments the current counter
        by the values stored in the second counter.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> a += b
        >>> a['first']
        1
        """
        for key, value in y.items():
            self[key] += value

    def __add__(self, y):
        """
        Adding two counters gives a counter with the union of all keys and
        counts of the second added to counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a + b)['first']
        1
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] + y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = y[key]
        return addend

    def __sub__(self, y):
        """
        Subtracting a counter from another gives a counter with the union of all keys and
        counts of the second subtracted from counts of the first.

        >>> a = Counter()
        >>> b = Counter()
        >>> a['first'] = -2
        >>> a['second'] = 4
        >>> b['first'] = 3
        >>> b['third'] = 1
        >>> (a - b)['first']
        -5
        """
        addend = Counter()
        for key in self:
            if key in y:
                addend[key] = self[key] - y[key]
            else:
                addend[key] = self[key]
        for key in y:
            if key in self:
                continue
            addend[key] = -1 * y[key]
        return addend

# ----------------------------------------------------------------------


class OrderedSet(collections.MutableSet):

    def __init__(self, iterable=None):
        self.end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.map = {}                   # key --> [key, prev, next]
        if iterable is not None:
            self |= iterable

    def __len__(self):
        return len(self.map)

    def __contains__(self, key):
        return key in self.map

    def add(self, key):
        if key not in self.map:
            end = self.end
            curr = end[1]
            curr[2] = end[1] = self.map[key] = [key, curr, end]

    def discard(self, key):
        if key in self.map:
            key, prev, next = self.map.pop(key)
            prev[2] = next
            next[1] = prev

    def __iter__(self):
        end = self.end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def pop(self, last=True):
        if not self:
            raise KeyError('set is empty')
        key = self.end[1][0] if last else self.end[2][0]
        self.discard(key)
        return key

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, list(self))

    def __eq__(self, other):
        if isinstance(other, OrderedSet):
            return len(self) == len(other) and list(self) == list(other)
        return set(self) == set(other)


# ----------------------------------------------------------------------

class Node(object):

    def __init__(self, data, p, n):
        self.data = data
        self.p = p  # previous
        self.n = n  # next

    def __repr__(self):
        pr, ne = False, False
        if self.p is not None:
            pr = '<{0}>'.format(self.p.data)
        if self.n is not None:
            ne = '<{0}>'.format(self.n.data)
        return '<{0},{1},{2}>'.format(pr, self.data, ne)


class DoubleList(object):
    head = None  # start of list
    tail = None  # end of list, where append

    def __init__(self, init_list=None):
        if init_list is not None:
            for data in init_list:
                self.append(data)

    def append(self, data):
        """
        Append to tail

        n1   <-> n2 <-> n3 <-> n4
        head                   tail

        :param data:
        :return:
        """
        new_node = Node(data, None, None)
        if self.head is None:
            self.head = self.tail = new_node
        else:
            new_node.p = self.tail
            new_node.n = None
            self.tail.n = new_node
            self.tail = new_node
        return new_node

    def remove(self, node):
        if self.head == node:
            self.head = node.n
        if self.tail == node:
            self.tail = node.p
        if node.p is not None:
            node.p.n = node.n
        if node.n is not None:
            node.n.p = node.p
        node.p = None
        node.n = None

    def find(self, data):
        node_list = list()
        current_node = self.head
        while current_node is not None:
            if current_node.data == data:
                node_list.append(current_node)
            current_node = current_node.n
        return node_list

    def to_list(self):
        data_list = list()
        current_node = self.head
        while current_node is not None:
            data_list.append(current_node.data)
            current_node = current_node.n
        return data_list

    def show(self):
        print 'DoubleList:'
        i = 0
        current_node = self.head
        while current_node is not None:
            print '{0} {1}'.format(i, current_node)
            current_node = current_node.n
            i += 1
        print '----------'


def test_double_list():

    print '\nCreate empty DoubleList...'
    dl1 = DoubleList()
    dl1.show()

    print '\nTry to remove 3 (get empty list)...'
    print 'dl1.find(3)', dl1.find(3)

    print '\nCreate from list [1, 2, 3, 4] ...'
    dl2 = DoubleList([1, 2, 3, 4])
    dl2.show()
    print 'dl2.to_list()', dl2.to_list()

    print '\nRemoving 3...'
    n = dl2.find(3)
    print 'dl2.find(3)', n
    n = n[0]
    dl2.remove(n)
    print 'n', n
    dl2.show()
    print 'dl2.to_list()', dl2.to_list()

    print '\nRemoving 1 (tests removing from head)...'
    n = dl2.find(1)
    print 'dl2.find(1)', n
    n = n[0]
    dl2.remove(n)
    print 'n', n
    dl2.show()
    print 'dl2.to_list()', dl2.to_list()

    print '\nRemoving 4 (tests removing from tail)...'
    n = dl2.find(4)
    print 'dl2.find(4)', n
    n = n[0]
    dl2.remove(n)
    print 'n', n
    dl2.show()
    print 'dl2.to_list()', dl2.to_list()

    print '\nRemoving 2 (tests removing last item)...'
    n = dl2.find(2)
    print 'dl2.find(2)', n
    n = n[0]
    dl2.remove(n)
    print 'n', n
    dl2.show()
    print 'dl2.to_list()', dl2.to_list()

# test_double_list()
