/* $Id: known_transition_matrix.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file known_transition_matrix.cpp
 *
 * @author Colin Dawson 
 */

#include "known_transition_matrix.h"
#include "util.h"
#include <m_cpp/m_matrix.h>
#include <m_cpp/m_vector.h>
#include <boost/make_shared.hpp>

Transition_prior_ptr Known_transition_parameters::make_module() const
{
    return boost::make_shared<Known_transition_matrix>(this);
}

void Known_transition_matrix::initialize_params()
{
    /*
     Transition matrix pi0_ and pi_ are both fixed over all iterations here
     either by using uniform probability amont all state or using the groundtruth transition matrix
     No update for transition model here
     */
    Base_class::initialize_params();
    if(uniform_initial_distribution)
    {
        pi0_ = kjb::log_normalize(Prob_vector((int) J(), 1.0));
    } else {
        PMWP(verbose_ > 0, "    Using fixed transition matrix from file %s and initial distributiuon \
            from file %s\n", (transition_matrix_path.c_str())(initial_distribution_path.c_str()));
        //std::cerr << "    Using fixed transition matrix from file " << transition_matrix_path
        //          << " and initial distribution from file " << initial_distribution_path
        //          << std::endl;
        Prob_vector tmp((initial_distribution_path + "ground_truth.txt").c_str());
        const size_t size = tmp.size();
        IFTD(size == J(),
             kjb::IO_error,
             "I/O ERROR: Reading in a size %d initial distribution, but model has J = %d.",
             (size)(J())
             );
        pi0_ = kjb::log_normalize(kjb::ew_log(tmp));
        if (verbose_ > 0) {std::cerr << "pi0 = " << kjb::ew_exponentiate(pi0()) << std::endl;}
    }
    if(uniform_transitions)
    {
        pi_ = kjb::log_normalize_rows(Prob_matrix((int) J(), (int) J(), 1.0));
    } else {
        Prob_matrix tmp((transition_matrix_path + "ground_truth.txt").c_str());
        const size_t rows = tmp.get_num_rows();
        const size_t cols = tmp.get_num_cols();
        IFTD(rows == J() && cols == J(),
             kjb::IO_error,
             "I/O ERROR: Reading in a %d by %d transition matrix, but model has J = %d.",
             (rows)(cols)(J())
            );
        pi_ = kjb::log_normalize_rows(kjb::ew_log(tmp));
        if (verbose_ > 0) {std::cerr << "pi = " << kjb::ew_exponentiate(pi()) << std::endl;}
    }
}

void Known_transition_matrix::input_previous_results(const std::string& input_path, const std::string& name)
{
    PM(verbose_ > 0, "Inputting information for known transition model...\n");
    //std::cerr << "Inputting information for known transition model..." << std::endl;
    Base_class::input_previous_results(input_path, name);
}

void Known_transition_matrix::set_up_results_log() const
{
    create_directory_if_nonexistent(write_path + "pi");
    create_directory_if_nonexistent(write_path + "pi0");
}

void Known_transition_matrix::write_state_to_file(const std::string&) const
{
    std::ofstream ofs;
    ofs.open(write_path + "pi0/" + "ground_truth.txt", std::ofstream::out);
    ofs << kjb::ew_exponentiate(pi0_) << " ";
    ofs.close();
    ofs.open(write_path + "pi/" + "ground_truth.txt", std::ofstream::out);
    ofs << ew_exponentiate(pi_) << " ";
    ofs.close();
}

