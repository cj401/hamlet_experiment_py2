/* $Id: probit_noise_model.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file probit_noise_model.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "probit_noise_model.h"
#include "emission_model.h"
#include <prob_cpp/prob_distribution.h>
#include <prob_cpp/prob_sample.h>
#include <prob_cpp/prob_pdf.h>
#include <boost/lexical_cast.hpp>
#include <boost/make_shared.hpp>
#include <limits>
#include <iostream>
#include <string>

/* /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ */ 

/*
Noise_model_ptr Probit_noise_parameters::make_module() const
{
    return boost::make_shared<Probit_noise_model>();
}
 */

Noise_model_ptr Probit_noise_parameters::make_module() const
{
    return boost::make_shared<Probit_noise_model>(this);
}

/* /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ */

void Probit_noise_model::initialize_resources()
{
    Base_class:initialize_resources();
    PM(verbose_ > 0, "       Allocating Y*...");
    //std::cerr << "        Allocating Y*...";
    Ystar_ = Latent_data_matrix_list(NF());
    for (size_t i = 0; i < NF(); i++)
    {
        Ystar_.at(i) = Latent_data_matrix(T(i), K(), 0.0);
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

/*
void Probit_noise_model::initialize_resources()
{
    Base_class::initialize_resources();
    std::cerr << "        Allocating Y*...";
    Ystar_ = Latent_data_matrix(T(), K(), 0.0);
    std::cerr << "done." << std::endl;
}
 */

void Probit_noise_model::initialize_params()
{
    Base_class::initialize_params();
    PM(verbose_ > 0, "       Initializng Y*...\n");
    //std::cerr << "        Initializing Y*..." << std::endl;
    for (size_t i = 0; i < NF(); i++)
    {
        if (Y_.at(i).get_num_rows() > 0) update_Ystar_(i);
    }
    PM(verbose_ > 0, "        Y* initialized.\n");
    //std::cerr << "        Y* initialized." << std::endl;
}

/*
void Probit_noise_model::initialize_params()
{
    Base_class::initialize_params();
    std::cerr << "        Initializing Y*..." << std::endl;
    if(Y_.get_num_rows() > 0) update_Ystar_();
    std::cerr << "        Y* initialized." << std::endl;
}
 */

void Probit_noise_model::input_previous_results(const std::string& input_path, const std::string& name)
{
    PMWP(verbose_ > 0, "Inpuuting information from iteration %s for probit noise model...\n", (name.c_str()));
    //std::cerr << "Inputting information from iteration " << name << " for probit noise model..." << std::endl;
    PM(verbose_ > 0, "Inputting information for latent data matrix list Ystar...\n");
    //std::cerr << "Inputting information for latent data matrix list Ystar..." << std::endl;
    if (NF() == 1)
    {
        PM(verbose_ > 0, "    Inputting information for Ystar0...\n");
        //std::cerr << "     Inputting information for Ystar0..." << std::endl;
        Ystar_[0] = Latent_data_matrix((input_path+"Ystar/"+name+".txt").c_str());
    }
    else
    {
        boost::format fmt("%03i");
        for (size_t i = 0; i < NF(); i++)
        {
            PMWP(verbose_ > 0, "    Inputting information for Ystar %d...", (i));
            //std::cerr << "     Inputting information for Ystar" << i << "..." << std::endl;
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            Ystar_[i] = Latent_data_matrix((input_path + "Ystar/" + name + "/" + file_name).c_str());
        }
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

void Probit_noise_model::add_data(
    const std::string& path,
    const size_t&      num_files
    )
{
    Y_ = Data_matrix_list(num_files);
    T_list T_t(num_files);
    if (num_files > 1)
    {
        boost::format fmt("%03i");
        for (size_t i = 0; i < num_files; i++)
        {
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            std::cerr << "Adding data from file " << path << "obs/" << file_name << std::endl;
            kjb::Matrix tmp;
            tmp.read((path + "obs/" + file_name).c_str());
            IFTD(tmp.get_num_rows() > 0, kjb::IO_error,
                 "I/O ERROR: Could not find data file %s.",
                 ((path + "obs/" + file_name).c_str()));
            IFTD(Y_.at(i).get_num_cols() == (int) Y_.at(0).get_num_cols(), kjb::IO_error,
                 "ERROR: Training data sets do not have consistent dimensionality %d, "
                 "data set %i has dimensionality %d",
                 (Y_.at(0).get_num_cols())(i+1)(Y_.at(i).get_num_cols()));
            Y_.at(i) = tmp.floor();
            T_t.at(i) = Y_.at(i).get_num_rows();
        }
    }
    else
    {
        PMWP(verbose_ > 0, "Adding data from file %sobs.txt\n", (path.c_str()));
        //std::cerr << "Adding data from file " << path << "obs.txt" << std::endl;
        kjb::Matrix tmp;
        tmp.read((path + "obs.txt").c_str());
        IFTD(tmp.get_num_rows() > 0, kjb::IO_error,
             "I/O ERROR: Could not find data file %s.",
             ((path + "obs.txt").c_str()));
        Y_.at(0) = tmp.floor();
        T_t.at(0) = Y_.at(0).get_num_rows();
    }
    set_data_size(T_t, Y_.at(0).get_num_cols());
    PM(verbose_ > 0, "    Allocating Y*...");
    //std::cerr << "    Allocating Y*...";
    Ystar_ = Latent_data_matrix_list(num_files);
    for (size_t i = 0; i < num_files; i++)
    {
        Ystar_.at(i) = Latent_data_matrix(T(i), K());
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

void Probit_noise_model::add_test_data(
    const std::string& path,
    const size_t&      num_files
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
            kjb::Matrix tmp;
            tmp.read((path + "obs/" + file_name).c_str());
            IFTD(tmp.get_num_rows() > 0, kjb::IO_error,
                 "I/O ERROR: Could not find data file %s.",
                 ((path + "obs.txt").c_str()));
            IFTD(tmp.get_num_cols() == (int) K(), kjb::IO_error,
                 "ERROR: Test data set has dimensionality %d, but training set has"
                 " dimensionality %d", (tmp.get_num_cols())(K()));
            Y_test_.at(i) = tmp.floor();
            test_T_t.at(i) = Y_test_.at(i).get_num_rows();
        }
    }
    else
    {
        PMWP(verbose_ > 0, "Adding test data from file %sobs.txt\n", (path.c_str()));
        //std::cerr << "Adding test data from file " << path << "obs.txt" << std::endl;
        kjb::Matrix tmp;
        tmp.read((path + "obs.txt").c_str());
        IFTD(tmp.get_num_rows() > 0, kjb::IO_error,
             "I/O ERROR: Could not find data file %s.",
             ((path + "obs.txt").c_str()));
        IFTD(tmp.get_num_cols() == (int) K(), kjb::IO_error,
             "ERROR: Test data set has dimensionality %d, but training set has"
             " dimensionality %d", (tmp.get_num_cols())(K()));
        Y_test_.at(0) = tmp.floor();
        test_T_t.at(0) = Y_test_.at(0).get_num_rows();
    }
    set_test_data_size(test_T_t);
    PM(verbose_ > 0, "Done adding test data.\n");
    //std::cerr << "Done adding test data." << std::endl;
}

// void Probit_noise_model::add_data(
//     const std::string& path
//     )
// {
//     std::cerr << "Adding data from file " << path << "obs.txt" << std::endl;
//     kjb::Matrix tmp;
//     tmp.read((path + "obs.txt").c_str());
//     IFTD(tmp.get_num_rows() > 0, kjb::IO_error,
//          "I/O ERROR: Could not find data file %s.",
//          ((path + "obs.txt").c_str()));
//     Y_ = tmp.floor();
//     set_data_size(Y_.get_num_rows(), Y_.get_num_cols());
//     std::cerr << "    Allocating Y*...";
//     Ystar_ = Latent_data_matrix(T(), K());
//     std::cerr << "done." << std::endl;
// }

//void Probit_noise_model::add_test_data(
//    const std::string& path
//    )
//{
//    std::cerr << "Adding test data from file " << path << "obs.txt" << std::endl;
//    kjb::Matrix tmp;
//    tmp.read((path + "obs.txt").c_str());
//    IFTD(tmp.get_num_rows() > 0, kjb::IO_error,
//         "I/O ERROR: Could not find data file %s.",
//         ((path + "obs.txt").c_str()));
//    IFTD(tmp.get_num_cols() == (int) K(), kjb::IO_error,
//         "ERROR: Test data set has dimensionality %d, but training set has"
//         " dimensionality %d", (Y_test_.get_num_cols())(K()));
//    tmp.write((write_path + "test_obs.txt").c_str());
//    Y_test_ = tmp.floor();
//    set_test_data_size(Y_test_.get_num_rows());
    // test_llm_ = Likelihood_matrix(test_T_ + 1, J(), 0.0);
//}

void Probit_noise_model::generate_data(
    const std::string& path
    )
{
    std::cerr << "    Allocating Y* and Y...";
    Ystar_ = Latent_data_matrix_list(NF());
    Y_ = Data_matrix_list(NF());
    for (size_t i = 0; i < NF(); i++)
    {
        Ystar_.at(i) = Latent_data_matrix(T(i), K(), 0.0);
        Y_.at(i) = Data_matrix(T(i), K());
        sample_data_from_prior_(Y_.at(i), Ystar_.at(i));
    }
    std::cerr << "done." << std::endl;
    if (NF() == 1)
    {
        std::cerr << "Writing data to file " << path + "obs.txt" << std::endl;
        Y_[0].write((path + "obs.txt").c_str());
    }
    else
    {
        create_directory_if_nonexistent((path + "obs").c_str());
        boost::format fmt("%03i");
        for (size_t i = 0; i < NF(); i++)
        {
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            std::cerr << "    Writing observations to file " << path + "obs/" + file_name << std::endl;
            Y_[i].write((path + "obs/" + file_name).c_str());
        }
    }
}

void Probit_noise_model::generate_test_data(
    const std::string& path
    )
{
    Latent_data_matrix_list Ystar(test_NF());
    Y_test_ = Data_matrix_list(test_NF());
    for (size_t i = 0; i < test_NF(); i++)
    {
        Ystar.at(i) = Latent_data_matrix(test_T(i), K(), 0.0);
        Y_test_.at(i) = Data_matrix(test_T(i), K());
        sample_data_from_prior_(Y_test_.at(i), Ystar.at(i));
    }
    if (test_NF()==1)
    {
        std::cerr << "Writing data to file " << path + "obs.txt" << std::endl;
        Y_test_[0].write((path + "obs.txt").c_str());
    }
    else
    {
        create_directory_if_nonexistent((path + "obs").c_str());
        boost::format fmt("%03i");
        for (size_t i = 0; i < test_NF(); i++)
        {
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            std::cerr << "    Writing observations to file " << path + "obs/" + file_name << std::endl;
            Y_test_[i].write((path + "obs/" + file_name).c_str());
        }
    }
}

/*
void Probit_noise_model::generate_data(
    const std::string& path
    )
{
    std::cerr << "    Allocating Y*...";
    Ystar_ = Latent_data_matrix(T(), K(), 0.0);
    std::cerr << "done." << std::endl;
    std::cerr << "    Allocating Y...";
    Y_ = Data_matrix(T(), K());
    std::cerr << "done." << std::endl;
    sample_data_from_prior_(Y_, Ystar_);
    std::cerr << "Writing data to file " << path + "obs.txt" << std::endl;
    Y_.write((path + "obs.txt").c_str());
}
 */

/*
 void Probit_noise_model::generate_test_data(
 const std::string& path
 )
 {
 Latent_data_matrix Ystar = Latent_data_matrix(T(), K(), 0.0);
 std::cerr << "    Allocating Y_test...";
 Y_test_ = Data_matrix(test_T_, K());
 std::cerr << "done." << std::endl;
 sample_data_from_prior_(Y_test_, Ystar);
 std::cerr << "Writing data to file " << path + "obs.txt" << std::endl;
 Y_test_.write((path + "obs.txt").c_str());
 }
*/

void Probit_noise_model::update_params()
{
    Base_class::update_params();
    for (size_t i = 0; i < NF(); i++)
    {
        update_Ystar_(i);
    }
}

/*
void Probit_noise_model::update_params()
{
    Base_class::update_params();
    update_Ystar_();
}
 */

void Probit_noise_model::set_up_results_log() const
{
    create_directory_if_nonexistent(write_path + "Ystar");
    if(NF() > 1)
    {
        boost::format fmt("%03i");
        create_directory_if_nonexistent((write_path + "obs").c_str());
        for (size_t i = 0; i < NF(); i++)
        {
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            Y_.at(i).write((write_path + "obs/" + file_name).c_str());
        }
    } else {
        Y_.at(0).write((write_path + "obs.txt").c_str());
    }
}

void Probit_noise_model::write_state_to_file(const std::string& name) const
{
    if (NF() == 1)
    {
        Ystar_.at(0).write((write_path + "Ystar/" + name + ".txt").c_str());
    }
    else
    {
        create_directory_if_nonexistent((write_path + "Ystar/" + name).c_str());
        boost::format fmt("%03i");
        for (size_t i = 0; i < NF(); i++)
        {
            std::string file_name = (fmt % (i+1)).str() + ".txt";
            Ystar_[i].write(
                (write_path + "Ystar/" + name + "/" + file_name).c_str());
        }
    }
}

/*
void Probit_noise_model::write_state_to_file(const std::string& name) const
{
    Ystar_.write((write_path + "Ystar/" + name + ".txt").c_str());
}
 */

double Probit_noise_model::log_likelihood_ratio_for_state_change(
    const size_t&           i,
    const Mean_vector&      x,
    const Mean_vector&      delta,
    const kjb::Index_range& times
                                                                 )
{
    Latent_data_matrix Yj = Ystar_.at(i)(times, kjb::Index_range(true));
    Mean_vector midpoint = x + 0.5 * delta; // K-vec
    Latent_data_matrix log_ratios = kjb::shift_rows_by(Yj, -midpoint);
    log_ratios.ew_multiply_rows_by(delta); // T x K
    return sum_matrix_rows(log_ratios).sum_vector_elements();
}

/*
double Probit_noise_model::log_likelihood_ratio_for_state_change(
    const Mean_vector&      x,
    const Mean_vector&      delta,
    const kjb::Index_range& times
    )
{
    Latent_data_matrix Yj = Ystar_(times, kjb::Index_range(true));
    Mean_vector midpoint = x + 0.5 * delta; // K-vec
    Latent_data_matrix log_ratios = kjb::shift_rows_by(Yj, -midpoint);
    log_ratios.ew_multiply_rows_by(delta); // T x K
    return sum_matrix_rows(log_ratios).sum_vector_elements();
    // Mean_vector new_mean = x + delta;
    // Prob_vector p_left_old(K());
    // Prob_vector p_left_new(K());
    // for(size_t k = 0; k < K(); ++k)
    // {
    //     p_left_old[k] = kjb::cdf(kjb::STD_NORMAL, x[k]);
    //     p_left_new[k] = kjb::cdf(kjb::STD_NORMAL, new_mean[k]);
    // }
    // kjb::Matrix Y(Y_);
    // kjb::Vector Y1 = kjb::sum_matrix_rows(Y(times, kjb::Index_range(true)));
    // kjb::Vector Y0 = kjb::Vector((int) K(), (double) times.size()) - Y1;
    // Prob_vector left_ratio = kjb::ew_log(p_left_new) - kjb::ew_log(p_left_old);
    // Prob_vector right_ratio = kjb::ew_log(1.0 - p_left_new) - kjb::ew_log(1.0 - p_left_old);
    // Prob_vector log_ratios = left_ratio.ew_multiply(Y1) + right_ratio.ew_multiply(Y0);
    // double result = log_ratios.sum_vector_elements();
    // // std::cerr << "left_ratio = " << left_ratio << std::endl;
    // // std::cerr << "right_ratio = " << right_ratio << std::endl;
    // // std::cerr << "Y1 counts = " << Y1 << std::endl;
    // // std::cerr << "Y0 counts = " << Y0 << std::endl;
    // // std::cerr << "Y0 + Y1 = " << Y0 + Y1 << std::endl;
    // // std::cerr << "log_ratios = " << log_ratio << std::endl;
    // return result;
}
 */

double Probit_noise_model::log_likelihood(
    const size_t&      i,
    const size_t&      t,
    const Mean_vector& mean
    ) const
{
    return -0.5 * vector_distance_squared(Ystar_.at(i).get_row(t), mean);
}

/*
double Probit_noise_model::log_likelihood(
    const size_t&      t,
    const Mean_vector& mean
    ) const
{
    return -0.5 * vector_distance_squared(Ystar_.get_row(t), mean);
    // Prob result = 0.0;
    // for(size_t k = 0; k < K(); ++k)
    // {
    //     Prob cum_prob = kjb::cdf(kjb::STD_NORMAL, mean[k]);
    //     result += Y(t,k) == 1 ? log(cum_prob) : log(1 - cum_prob);
    // }
    // return result;
}

double Probit_noise_model::test_log_likelihood(
    const size_t&      t,
    const Mean_vector& mean
    ) const
{
 // return -0.5 * vector_distance_squared(Ystar_.get_row(t), mean);
    Prob result = 0.0;
    for(size_t k = 0; k < K(); ++k)
    {
        Prob cum_prob = kjb::cdf(kjb::STD_NORMAL, mean[k]);
        result += Y_test_(t,k) == 1 ? log(cum_prob) : log(1 - cum_prob);
    }
    return result;
}
 */

double Probit_noise_model::test_log_likelihood(
    const size_t&      i,
    const size_t&      t,
    const Mean_vector& mean
    ) const
{
    // return -0.5 * vector_distance_squared(Ystar_.get_row(t), mean);
    Prob result = 0.0;
    for(size_t k = 0; k < K(); ++k)
    {
        Prob cum_prob = kjb::cdf(kjb::STD_NORMAL, mean[k]);
        result += Y_test_[i](t,k) == 1 ? log(cum_prob) : log(1 - cum_prob);
    }
    return result;
}

void Probit_noise_model::update_Ystar_(const size_t& i)
{
    // resample Ystar_
    PM(verbose_ > 0, "    Updating Y*...");
    //std::cerr << "    Updating Y*... ";
    const Mean_matrix_list& Xstar = parent->X_star();
    for(size_t t = 0; t < T(i); ++t)
    {
        for(size_t k = 0; k < K(); ++k)
        {
            double mean = Xstar.at(i)(t, k);
            if(Y(i,t,k) == 1) Ystar(i,t,k) = sample_left_truncated_normal(mean);
            else Ystar(i,t,k) = sample_right_truncated_normal(mean);
        }
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

void Probit_noise_model::update_Ystar_()
{}

/*
void Probit_noise_model::update_Ystar_()
{
    // resample Ystar_
    std::cerr << "    Updating Y*... ";
    const Mean_matrix& Xstar = parent->X_star();
    for(size_t t = 0; t < T(); ++t)
    {
        for(size_t k = 0; k < K(); ++k)
        {
            double mean = Xstar(t, k);
            if(Y(t,k) == 1) Ystar(t,k) = sample_left_truncated_normal(mean);
            else Ystar(t,k) = sample_right_truncated_normal(mean);
        }
    }
    std::cerr << "done." << std::endl;
}
 */

/*
 void Probit_noise_model::sample_data_from_prior_(
 Data_matrix& observed,
 Latent_data_matrix& latent
 )
 {
 // generate Ystar_ given weights
 kjb::Matrix cov = kjb::create_identity_matrix(K());
 const kjb::Matrix& Xstar = parent->X_star();
 std::cerr << "    Sampling Y* from prior...";
 sample_mv_normal_vectors_from_row_means(latent, Xstar.at(0), cov);
 std::cerr << "done." << std::endl;
 std::cerr << "    Truncating Y from Y*..";
 for(size_t k = 0; k < K(); ++k)
 {
 for(size_t t = 0; t < T(); ++t)
 {
 observed.at(t,k) = latent.at(t,k) > 0 ? 1 : 0;
 }
 }
 std::cerr << "done." << std::endl;
 }
 */

void Probit_noise_model::sample_data_from_prior_(
    Data_matrix& observed,
    Latent_data_matrix& latent
    )
{
    // generate Ystar_ given weights
    kjb::Matrix cov = kjb::create_identity_matrix(K());
    const std::vector<kjb::Matrix>& Xstar = parent->X_star();
    PM(verbose_ > 0, "    Sampling Y* from prior...");
    //std::cerr << "    Sampling Y* from prior...";
    sample_mv_normal_vectors_from_row_means(latent, Xstar.at(0), cov);
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
    PM(verbose_ > 0, "    Truncating Y from Y*...");
    //std::cerr << "    Truncating Y from Y*..";
    for(size_t k = 0; k < K(); ++k)
    {
        for(size_t t = 0; t < T(0); ++t)
        {
            observed.at(t,k) = latent.at(t,k) > 0 ? 1 : 0;
        }
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

/*
void Probit_noise_model::sample_data_from_prior_(
    Data_matrix_list& observed,
    Latent_data_matrix_list& latent
    )
{
    // generate Ystar_ given weights
    kjb::Matrix cov = kjb::create_identity_matrix(K());
    const std::vector<kjb::Matrix>& Xstar = parent->X_star();
    std::cerr << "    Sampling Y* from prior...";
    for (size_t i = 0; i < NF(); i++)
    {
        sample_mv_normal_vectors_from_row_means(latent.at(i), Xstar.at(i), cov);
    }
    std::cerr << "done." << std::endl;
    std::cerr << "    Truncating Y from Y*..";
    for (size_t i = 0; i < NF(); i++)
    {
        for(size_t k = 0; k < K(); ++k)
        {
            for(size_t t = 0; t < T(i); ++t)
            {
                observed.at(i).at(t,k) = latent.at(i).at(t,k) > 0 ? 1 : 0;
            }
        }
    }
    std::cerr << "done." << std::endl;
}
 */

