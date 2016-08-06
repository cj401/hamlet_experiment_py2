import multiprocessing       # provides: Pool, Lock, cpu_count, map
from subprocess import call  # used to call command-line program
import os                    # provides: getcwd, chdir
import random  # just used to generate random payloads


lock = None  # global var to store lock


class FunctionSpec:
    """
    Class used to store parameters for my_function and my_function_to_make_command_line_call
    """

    def __init__(self, num, payload,
                 main_path='../'  # suppose the command-line fn is one directory up...
                 ):
        self.num = num
        self.payload = payload
        self.main_path = main_path

    def pprint(self):
        print 'spec num={0}, payload={1}, main_path=\'{2}\''\
            .format(self.num, self.payload, self.main_path)


def my_function(spec):
    global lock  # python idiom: must declare up front any vars that you are treating as global

    """
    Example function to call with multiprocessing
    :param spec:
    :return:
    """
    print "[{0}] May be overwritten by multiple processes".format(spec.num)

    lock.acquire()  # get the lock
    # While you have the lock, this only this process gets to do stuff here...
    # This is a good place to do things like write to a log file
    print ">>[{0}] Protected from being overwritten ({1})<<".format(spec.num, spec.payload)
    lock.release()  # release the lock

    # NOTE: There are more complicated ways, including special figures structures,
    # for sharing info between processes.  I haven't explored that functionality.

    # returning this information allows you to keep track of which spec run
    # this was, and what the process return value was (whether is succeeded or not)
    return spec.num, 0  # here 0 means "success"


def my_function_to_make_command_line_call(spec):
    """
    Example function that uses subprocess.call to call a command-line function.

    Assume the command-line function is called as follows:
        $ my_cli_function -a <arg1> -b <arg2>

    The following is NOT necessary, but I've found to be a helpful pattern:
    Suppose your command-line function needed to be called from a directory
    that is _not_ the same as the one you call the outer python run_experiment_script.
    In that case, use the spec.main_path to switch the python current working
    directory (cwd) from it's current location to spec.main_path, then make
    the command-line call, then switch back to the original cwd.
    In the FunctionSpec class, the default for main_path is "one directory up".

    :param spec:
    :return:
    """

    # build up the command-line command that will be called
    command = './my_cli_function'
    command += ' -a {0}'.format(spec.num)
    command += ' -b {0}'.format(spec.payload)

    # save original working directory from which python was called
    owd = os.getcwd()  # get's current working directory
    # change the current working directory to spec.main_path
    os.chdir(spec.main_path)

    # this executes the command-line call
    # the return value is the unix command exit value, as though it were
    # run from the cli.  Usually 0 means success, -1 means failure
    ret = call(command, shell=True)

    # change the current working directory back to owd.
    os.chdir(owd)

    # returning this information allows you to keep track of which spec run
    # this was, and what the process return value was (whether is succeeded or not)
    return spec.num, ret


def run_multiprocessing(verbose=True):
    global lock  # python idiom: must declare up front any vars that you are treating as global

    # Create the lock that will be shared during this run
    # The lock is a SINGLE token that when possessed, is unavailable for anyone else to use.
    lock = multiprocessing.Lock()

    # could manually specify the number of cpu's available
    # cpu_count() determines the total cpus available on your machine
    processor_pool_size = multiprocessing.cpu_count()

    # create the process pool: how many cpus are available for jobs
    p = multiprocessing.Pool(processor_pool_size)

    # Here we will use special multiprocessing map function that will map
    # across a "parameters_spec_list".
    # The parameter_spec_list is a list the contains all of the parameter
    # setting for all of the calls we want to make to my_function;
    # These parameter setting have to be determined ahead of time before
    # making the calls in the map function.
    # The map function requires that my_function take only ONE argument,
    # but often we want multiple parameters for each call.
    # To work around this restriction, we can create functions that receive
    # multiple parameters either in a primitive figures structure (e.g., list,
    # set, dict), or as a class.  Here we use the 'FunctionSpec' class
    # to store our parameters for each call to my_function.
    # The following builds up a list of 50 different specs (sets of parameters)
    # to call my_function with (I'm using a python 'list comprehension' here
    # to build up the list).
    parameter_spec_list = [ FunctionSpec(i, random.random()) for i in range(50) ]

    # display the parameter specs
    if verbose:
        print '\nparameter_spec_list:\n'
        for spec in parameter_spec_list:
            spec.pprint()
        print '\nNow run the processes (NOTE: not guaranteed to display in order):\n'

    # NOTE: Most of what is going on in run_experiments.py is building up
    # the parameter_spec_list.
    # AND, just to be clear, parameter_spec_list could be named anything you want,
    # the map function only cares that it is something iterable...

    # Make call to Pool.map:
    # This works like a functional map idiom, but assumes multiple
    # cpus available for each call to my_function.  It works as if mapping
    # my_function across the parameter_spec_list.
    # HOWEVER, note that the order in which calls to my_function are made is
    # controlled by the OS and cpu load, so behavior in each call is NOT
    # guaranteed to be physically executed in the order of processor_pool_size;
    # and as soon as a cpu is freed, the next call to my_function is made.
    # ON THE OTHER HAND, just like a standard functional map idiom, results
    # WILL be stored in same order as parameter_spec_list, as if my_function were
    # iterated over the list in that order.  To achieve this, p.map won't
    # 'complete' execution and return until after all of the subprocess calls
    # to my_function have completed.
    results = p.map(my_function, parameter_spec_list)

    # With enough processes (above I create 50), you may see in the output
    # that some of the print statements in my_function are displayed in
    # different order than they were specified.  I had written it so that
    # some of the text could in principle be overwritten in the display,
    # although I can't seem to get it to actually overwrite -- and that
    # might be b/c those functions are within python;
    # however, I have definitely seen overwriting of text happen
    # when calling more involved command-line fns like hamlet.

    print '\n(NOTE that results are in order...)\nresults:', results


# -----------------------------------------------------------------------------
# script

# execute the top-level command
run_multiprocessing()
