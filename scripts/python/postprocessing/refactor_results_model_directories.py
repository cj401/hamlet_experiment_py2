import os
import shutil
import sys

from utilities import util

__author__ = 'clayton'

'''
Script to refactor results directory branches, to combine old-style model spec
that split model spec from LT/noLT so that now LT/noLT is _part_ of model spec.
Format:
   <data_spec>/<model_spec>/<replication_number>

For example, previous branch:
   results/cocktail/h10.0_nocs_cp1/hsmm_hdp_w1/noLT_10
now becomes:
   results/cocktail/h10.0_nocs_cp1/hsmm_hdp_w1_noLT/10

This run_experiment_script likely only needs to be run once.
'''

def get_src_dst_pairs(results_root, main_path='../'):
    branches = util.get_directory_branches(root_dir=results_root, main_path=main_path)
    branch_pairs = list()
    for branch in branches:
        branch_parts = branch.split('/')
        model_component = branch_parts[-2]
        replication_component = branch_parts[-1]
        split_p = True
        if 'BFact' in model_component.split('_'):
            split_p = False
        model_component_parts = model_component.split('_')
        replication_component_parts = replication_component.split('_')
        new_model_component_parts = model_component_parts
        if split_p:
            new_model_component_parts += replication_component_parts[:-1]
        new_model_component = ['_'.join(new_model_component_parts)]
        new_replication_component = [replication_component_parts[-1]]
        new_branch_parts = branch_parts[:-2] + new_model_component + new_replication_component
        new_branch = '/'.join(new_branch_parts)
        branch_pairs.append( (branch, new_branch, split_p) )
    return branch_pairs


def combine_model_description_directories(results_root, main_path='../', test_p=True):
    branch_pairs = get_src_dst_pairs(results_root, main_path)

    owd = os.getcwd()
    os.chdir(main_path)

    for (src, dst, split_p) in branch_pairs:

        print

        if split_p:
            # create destination_path
            destination_path = '/'.join(dst.split('/')[:-1])

            if not test_p:
                if os.path.exists(destination_path):
                    print 'NOTE: Path already exists: \'{0}\''\
                        .format(destination_path)
                else:
                    os.makedirs(destination_path)
                    print 'Created destination_path: \'{0}\''\
                        .format(destination_path)
            else:
                print 'Would create destination_path: \'{0}\''\
                    .format(destination_path)

        if not test_p:
            # move
            shutil.move(src, dst)
            print 'Moved \'{0}\' to \'{1}\''.format(src, dst)
        else:
            print 'Would move \'{0}\' to \'{1}\''.format(src, dst)

        if split_p:
            # remove source_parent_path
            source_parent_path = '/'.join(src.split('/')[:-1])

            if not test_p:
                try:
                    os.rmdir(source_parent_path)
                    print '**** Removed source_parent_path: \'{0}\''\
                        .format(source_parent_path)
                except OSError as ex:
                    if ex.errno == os.errno.ENOTEMPTY:
                        print 'NOTE: Directory not empty: \'{0}\''\
                            .format(source_parent_path)
                    else:
                        print ex
                        sys.exit()
            else:
                print 'Would remove source_parent_path: \'{0}\''\
                    .format(source_parent_path)

    os.chdir(owd)
    print '\nDONE'

# combine_model_description_directories('results/cocktail', test_p=True)
