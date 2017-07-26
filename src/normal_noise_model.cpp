/* $Id: normal_noise_model.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file normal_noise_model.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "normal_noise_model.h"
#include "parameters.h"
#include <m_cpp/m_vector.h>
#include <m_cpp/m_matrix.h>
#include <m_cpp/m_mat_view.h>
#include <prob_cpp/prob_distribution.h>
#include <prob_cpp/prob_sample.h>
#include <boost/make_shared.hpp>
#include <string>
#include <iostream>

/*
Normal_noise_parameters::Normal_noise_parameters(
    const Parameters& params,
    const std::string& name
    ) : a_h(params.get_param_as<double>(
                name, "a_h", bad_h_prior(), valid_shape_param)),
        b_h(params.get_param_as<double>(
                name, "b_h", bad_h_prior(), valid_rate_param))
{}
 */

Normal_noise_parameters::Normal_noise_parameters(
    const Parameters& params,
    const std::string& name
    ) : Noise_model_parameters(params, name),
        a_h(params.get_param_as<double>(
                name, "a_h", bad_h_prior(), valid_shape_param)),
        b_h(params.get_param_as<double>(
                name, "b_h", bad_h_prior(), valid_rate_param))
{}

Noise_model_ptr Normal_noise_parameters::make_module() const
{
    return boost::make_shared<Normal_noise_model>(this);
}

void Normal_noise_model::initialize_resources()
{
    Base_class::initialize_resources();
    h_ = Precision_vector(K());
}

void Normal_noise_model::initialize_params()
{
    Base_class::initialize_params();
    initialize_h_from_prior_();
}

void Normal_noise_model::input_previous_results(const std::string& input_path, const std::string& name)
{
    PMWP(verbose_ > 0, "Inputting information from iteration %s for normal noise model...\n", (name.c_str()));
    //std::cerr << "Inpurtting information from iteration " << name << " for normal_noise_model..." << std::endl;
    PM(verbose_ > 0, "Inputting information for precision vector h_...\n");
    //std::cerr << "Inputting infromation for precision vector h_..." << std::endl;
    h_ = Precision_vector(input_to_vector<double>(input_path, "h.txt", name));
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done!" << std::endl;
}

void Normal_noise_model::add_data(const std::string& path, const size_t& num_files)
{
    PM(verbose_ > 0, "     Initializing data matrix list...");
    //std::cerr << "    Initializing data matrix list...";
    Y_ = Data_matrix_list(num_files);
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
    T_list T_t(num_files);
    if (num_files > 1)
    {
        boost::format fmt("%03i");
        for (size_t i = 0; i < num_files; i++)
        {
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            std::cerr << "Adding data from file " << path << "obs/" << file_name << std::endl;
            Y_.at(i).read((path + "obs/" + file_name).c_str());
            IFTD(Y_.at(i).get_num_rows() > 0, kjb::IO_error,
                 "I/O ERROR: Could not find data file %s.",
                 ((path + "obs/" + file_name).c_str()));
            IFTD(Y_.at(i).get_num_cols() == (int) Y_.at(0).get_num_cols(), kjb::IO_error,
                 "ERROR: Training data sets do not have consistent dimensionality %d, "
                 "data set %i has dimensionality %d",
                 (Y_.at(0).get_num_cols())(i+1)(Y_.at(i).get_num_cols()));
            T_t.at(i) = Y_.at(i).get_num_rows();
        }
    }
    else
    {
        std::cerr << "Adding data from file " << path << "obs.txt" << std::endl;
        Y_.at(0).read((path + "obs.txt").c_str());
        IFTD(Y_.at(0).get_num_rows() > 0, kjb::IO_error,
             "I/O ERROR: Could not find data file %s.",
             ((path + "obs.txt").c_str()));
        T_t.at(0) = Y_.at(0).get_num_rows();
    }
    set_data_size(T_t, Y_.at(0).get_num_cols());
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

void Normal_noise_model::add_test_data(
        const std::string& path,
        const size_t& num_files
        )
{
    Y_test_ = Data_matrix_list(num_files);
    T_list test_T_t(num_files);
    if (num_files > 1)
    {
        boost::format fmt("%03i");
        for (size_t i = 0; i < num_files; i++)
        {
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            std::cerr << "Adding data from file " << path << "obs/" << file_name << std::endl;
            Y_test_.at(i).read((path + "obs/" + file_name).c_str());
            IFTD(Y_test_.at(i).get_num_rows() > 0, kjb::IO_error,
                 "I/O ERROR: Could not find data file %s.",
                 ((path + "obs/" + file_name).c_str()));
            IFTD(Y_test_.at(i).get_num_cols() == (int) K(), kjb::IO_error,
                 "ERROR: Test data set %s has dimensionality %d, but training set has"
                 " dimensionality %d", (file_name.c_str())(Y_test_.at(i).get_num_cols())(K()));
            test_T_t.at(i) = Y_test_.at(i).get_num_rows();
        }
    }
    else
    {
        std::cerr << "Adding test data from file " << path << "obs.txt" << std::endl;
        Y_test_.at(0).read((path + "obs.txt").c_str());
        IFTD(Y_test_.at(0).get_num_rows() > 0, kjb::IO_error,
             "I/O ERROR: Could not find data file %s.",
             ((path + "obs.txt").c_str()));
        IFTD(Y_test_.at(0).get_num_cols() == (int) K(), kjb::IO_error,
             "ERROR: Test data set has dimensionality %d, but training set has"
             " dimensionality %d", (Y_test_.at(0).get_num_cols())(K()));
        test_T_t.at(0) = Y_test_.at(0).get_num_rows();
    }
    set_test_data_size(test_T_t);
    //write test data to result directory?
}

// void Normal_noise_model::add_data(
//     const std::string& path
//     )
// {
//     std::cerr << "Adding data from file " << path << "obs.txt" << std::endl;
//     Y_.read((path + "obs.txt").c_str());
//     IFTD(Y_.get_num_rows() > 0, kjb::IO_error, 
//          "I/O ERROR: Could not find data file %s.",
//          ((path + "obs.txt").c_str()));
//     set_data_size(Y_.get_num_rows(), Y_.get_num_cols());
// }

//void Normal_noise_model::add_test_data(
//    const std::string& path
//    )
//{
//    std::cerr << "Adding test data from file " << path << "obs.txt" << std::endl;
//    Y_test_.read((path + "obs.txt").c_str());
//    IFTD(Y_test_.get_num_rows() > 0, kjb::IO_error,
//         "I/O ERROR: Could not find data file %s.",
//         ((path + "obs.txt").c_str()));
//    IFTD(Y_test_.get_num_cols() == (int) K(), kjb::IO_error,
//         "ERROR: Test data set has dimensionality %d, but training set has"
//         " dimensionality %d", (Y_test_.get_num_cols())(K()));
//    set_test_data_size(Y_test_.get_num_rows());
    // test_llm_ = Likelihood_matrix(test_T_ + 1, J(), 0.0);
//    Y_test_.write((write_path + "test_obs.txt").c_str());
//}

/*
void Normal_noise_model::generate_data(
    const std::string& path
    )
{
    std::cerr << "Generating data..." << std::endl;
    std::cerr << "    Allocating Y...";
    Y_ = Data_matrix(T(), K());
    std::cerr << "done." << std::endl;
    sample_data_from_prior_(Y_);
    std::cerr << "    Writing observations to file " << path + "obs.txt" << std::endl;
    Y_.write((path + "obs.txt").c_str());
}
 */

void Normal_noise_model::generate_data(
    const std::string& path
    )
{
    std::cerr << "Generating data..." << std::endl;
    std::cerr << "    Allocating Y...";
    Y_ = Data_matrix_list(NF());
    for (size_t i = 0; i < NF(); i++)
    {
        Y_.at(i) = Data_matrix(T(i), K());
        sample_data_from_prior_(Y_.at(i));
    }
    if (NF() == 1)
    {
        std::cerr << "    Writing observations to file " << path + "obs.txt" << std::endl;
        Y_.at(0).write((path + "obs.txt").c_str());
    }
    else
    {
        create_directory_if_nonexistent((path + "obs").c_str());
        boost::format fmt("%03i");
        for (size_t i = 0; i < NF(); i++)
        {
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            std::cerr << "    Writing observations to file " << path + "obs/" + file_name << std::endl;
            Y_.at(i).write((path + "obs/" + file_name).c_str());
        }
    }
}

void Normal_noise_model::generate_test_data(
    const std::string& path
    )
{
    std::cerr << "Generating test data..." << std::endl;
    Y_test_ = Data_matrix_list(test_NF());
    for (size_t i = 0; i < test_NF(); i++)
    {
        Y_test_.at(i) = Data_matrix(test_T(i), K());
        sample_data_from_prior_(Y_test_.at(i));
    }
    if (test_NF() == 1)
    {
        std::cerr << "    Writing observations to file " << path + "obs.txt" << std::endl;
        Y_test_.at(0).write((path + "obs.txt").c_str());
    }
    else
    {
        create_directory_if_nonexistent((path + "obs").c_str());
        boost::format fmt("%03i");
        for (size_t i = 0; i < test_NF(); i++)
        {
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            std::cerr << "    Writing observations to file " << path + "obs/" + file_name << std::endl;
            Y_test_.at(i).write((path + "obs/" + file_name).c_str());
        }
    }
}

/*
void Normal_noise_model::generate_test_data(
    const std::string& path
    )
{
    std::cerr << "Generating test data..." << std::endl;
    Y_test_ = Data_matrix(test_T_, K());
    sample_data_from_prior_(Y_test_);
    std::cerr << "    Writing observations to file " << path + "obs.txt" << std::endl;
    Y_test_.write((path + "obs.txt").c_str());
}
 */

void Normal_noise_model::update_params()
{
    /*
     update covariance matrix for y, 
     refer to section 4.1 "sampling W and Sigma" in the paper
     */
    PM(verbose_ > 0, "    Updating h...");
    //std::cerr << "    Updating h...";
    const Data_matrix_list& X = parent->X_star();
    Data_vector sse = Data_vector((int)K(), 0.0);
    size_t sum_T = 0;
    for (size_t i = 0; i < NF(); i++)
    {
        sum_T = sum_T + T(i);
        sse =  sse + kjb::sum_matrix_rows(kjb::get_squared_elements(Y_.at(i) - X.at(i)));
    }
    for(size_t k = 0; k < K(); ++k)
    {
        Precision_dist r_h(a_h_ + sum_T / 2, 2.0 / (2*b_h_ + sse[k]));
        h_[k] = kjb::sample(r_h);
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

/*
void Normal_noise_model::update_params()
{
    std::cerr << "    Updating h...";
    const Data_matrix& X = parent->X_star();
    Data_vector sse = kjb::sum_matrix_rows(kjb::get_squared_elements(Y_ - X));
    for(size_t k = 0; k < K(); ++k)
    {
        Precision_dist r_h(a_h_ + T() / 2, 2.0 / (2*b_h_ + sse[k]));
        h_[k] = kjb::sample(r_h);
    }
    std::cerr << "done." << std::endl;
}
 */

void Normal_noise_model::set_up_results_log() const
{
    std::ofstream ofs;
    ofs.open(write_path + "h.txt", std::ofstream::out);
    ofs << "iteration value" << std::endl;
    ofs.close();
    if(NF() > 1)
    {
        boost::format fmt("%03i");
        create_directory_if_nonexistent((write_path + "obs/").c_str());
        for (size_t i = 0; i < NF(); i++)
        {
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            Y_.at(i).write((write_path + "obs/" + file_name).c_str());
        }
    } else {
        Y_.at(0).write((write_path + "obs.txt").c_str());
    }
}

void Normal_noise_model::write_state_to_file(const std::string& name) const
{
    std::ofstream ofs;
    ofs.open(write_path + "h.txt", std::ofstream::app);
    ofs << name << h_ << std::endl;
    ofs.close();
}

void Normal_noise_model::initialize_h_from_prior_()
{
    /*
     each dimension's precision is sample from a gamma distribution
     */
    PM(verbose_ > 0, "    Initializing h from prior...");
    //std::cerr << "    Initializing h from prior...";
    for(size_t k = 0; k < K(); ++k)
    {
        Precision_dist r_h(a_h_, 1.0 / b_h_);
        h_[k] = kjb::sample(r_h);
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

double Normal_noise_model::log_likelihood_ratio_for_state_change(
    const size_t&           i,
    const Mean_vector&      x,
    const Mean_vector&      delta,
    const kjb::Index_range& times
    )
{
    Data_matrix Yj = Y_.at(i)(times, kjb::Index_range(true));
    Mean_vector midpoint = x + 0.5 * delta; // K-vec
    Data_matrix log_ratios = kjb::shift_rows_by(Yj, -midpoint); // Tj x K
    log_ratios.ew_multiply_rows_by(delta); // Tj x K
    return sum_matrix_rows(log_ratios).sum_vector_elements(); // scalar
}

/*
double Normal_noise_model::log_likelihood_ratio_for_state_change(
    const Mean_vector&      x,
    const Mean_vector&      delta,
    const kjb::Index_range& times
    )
{
    Data_matrix Yj = Y_(times, kjb::Index_range(true));
    Mean_vector midpoint = x + 0.5 * delta; // K-vec
    Data_matrix log_ratios = kjb::shift_rows_by(Yj, -midpoint); // Tj x K
    log_ratios.ew_multiply_rows_by(delta); // Tj x K
    return sum_matrix_rows(log_ratios).sum_vector_elements(); // scalar
}
 */

double Normal_noise_model::log_likelihood(
    const size_t&      i,
    const size_t&      t,
    const Mean_vector& mean
    ) const
{
    return log_likelihood(Y_.at(i), t, mean, h_);
}

double Normal_noise_model::test_log_likelihood(
    const size_t&      i,
    const size_t&      t,
    const Mean_vector& mean
    ) const
{
    return log_likelihood(Y_test_.at(i), t, mean, h_);
}

/*
double Normal_noise_model::log_likelihood(
    const size_t&      t,
    const Mean_vector& mean
    ) const
{
    return log_likelihood(Y_, t, mean, h_);
}
double Normal_noise_model::test_log_likelihood(
    const size_t&      t,
    const Mean_vector& mean
) const
{
    return log_likelihood(Y_test_, t, mean, h_);
}
 */

double Normal_noise_model::log_likelihood(
    const Data_matrix&      data,
    const size_t&           t,
    const Mean_vector&      mean,
    const Precision_vector& h
    )
{
    return
         (0.5 * kjb::ew_log(1.0 / (2*M_PI) * h)
         - 0.5 * kjb::square_elements(data.get_row(t) - mean).ew_multiply(h)).sum_vector_elements();
}

/*
void Normal_noise_model::sample_data_from_prior_(Data_matrix_list& Y)
{
    const Mean_matrix_list& means = parent->X_star();
    kjb::Vector scales = kjb::ew_reciprocal(h_);
    kjb::Matrix cov = kjb::create_diagonal_matrix(scales);
    std::cerr << "    Sampling data from prior...";
    for (size_t i = 0; i < NF(); i++)
    {
        sample_mv_normal_vectors_from_row_means(Y.at(i), means.at(i), cov);
    }
    std::cerr << "done." << std::endl;
}
 */

void Normal_noise_model::sample_data_from_prior_(Data_matrix& Y)
{
    const Mean_matrix_list& means = parent->X_star();
    kjb::Vector scales = kjb::ew_reciprocal(h_);
    kjb::Matrix cov = kjb::create_diagonal_matrix(scales);
    PM(verbose_ > 0, "    Sampling data from prior...");
    //std::cerr << "    Sampling data from prior...";
    sample_mv_normal_vectors_from_row_means(Y, means.at(0), cov);
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

