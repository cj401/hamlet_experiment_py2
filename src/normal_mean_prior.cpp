/* $Id: normal_mean_prior.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file normal_mean_prior.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "normal_mean_prior.h"
#include "mean_emission_model.h"
#include <boost/make_shared.hpp>

Mean_prior_ptr Normal_mean_prior_params::make_module() const
{
    return boost::make_shared<Normal_mean_prior>(this);
}

Normal_mean_prior::Normal_mean_prior(const Params* const hyperparameters)
    : Base_class(hyperparameters),
      prior_precision_(hyperparameters->prior_precision)
{}

void Normal_mean_prior::initialize_resources()
{
    Base_class::initialize_resources();
}

void Normal_mean_prior::initialize_params()
{
    kjb::Matrix prior_covariance =
        kjb::create_diagonal_matrix(K(), 1.0 / prior_precision_);
    kjb::Vector prior_mean((int) K(), 0.0);
    kjb::MV_normal_distribution r_X(prior_mean, prior_covariance);
    for(size_t j = 0; j < J(); ++j)
    {
        X_.set_row(j, kjb::sample(r_X));
    }
}

void Normal_mean_prior::input_previous_results(const std::string& input_path, const std::string& name)
{
    PMWP(verbose_ > 0, "Inputting information from itreation %s for normal mean prior...\n", (name.c_str()));
    //std::cerr << "Inputting information from iteration " << name << " for normal mean prior..." << std::endl;
    PM(verbose_ > 0, "Inputting information for mean matrix X_...\n");
    //std::cerr << "Inputting information for mean matrix X_..." << std::endl;
    X_ = Mean_matrix((input_path + "X/" + name + ".txt").c_str());
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

void Normal_mean_prior::generate_data(const std::string&)
{
    initialize_params();
    PMWP(verbose_ > 0, "    Writing means to file %smeans.txt\n", (write_path.c_str()));
    //std::cerr << "    Writing means to file " << write_path + "means.txt" << std::endl;
    X_.write((write_path + "means.txt").c_str());
}

void Normal_mean_prior::update_params()
{
    PM(verbose_ > 0, "    Updating means X...\n");
    //std::cerr << "    Updating means X...";
    const Noisy_data_matrix_list& Y = noisy_data();
    Scale_vector h(noise_parameters());
    Mean_vector posterior_mean((int) K(), 0.0);
    Scale_vector posterior_precision((int) K(), 1.0 / prior_precision_);
    for (size_t j = 0; j < J(); ++j)
    {
        size_t n_j = 0;
        Mean_vector ysum_j = Mean_vector((int) K(), 0.0);
        for (size_t i = 0; i < NF(); ++i)
        {
            kjb::Index_range rows(partition_map(i,j));
            size_t n_ij = rows.size();
            if (n_ij > 0)
            {
                n_j = n_j + n_ij;
                kjb::Const_matrix_view Yi_j = Y.at(i)(rows, kjb::Index_range::ALL);
                ysum_j = ysum_j + kjb::sum_matrix_rows(Yi_j);
            }
        }
        if (n_j > 0)
        {
            Mean_vector ybar_j = ysum_j / n_j;
            posterior_precision = n_j * h + prior_precision_;
            posterior_mean =
                n_j * ybar_j.ew_multiply(h).ew_divide(posterior_precision);
        }
        for(size_t k = 0; k < K(); ++k)
        {
            Normal_dist r_x_jk(posterior_mean(k), 1.0 / posterior_precision(k));
            X_(j,k) = kjb::sample(r_x_jk);
        }
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

/*
void Normal_mean_prior::update_params()
{
    std::cerr << "    Updating means X...";
    const Noisy_data_matrix& Y = noisy_data();
    // std::cerr << "        Y is size " << Y.get_num_rows()
    //           << " x " << Y.get_num_cols() << std::endl;
    // std::cerr << "        H is size " << noise_parameters().get_num_rows()
    //           << " x " << noise_parameters().get_num_cols() << std::endl;
    Scale_vector h(noise_parameters());
    // std::cerr << "        h is size " << h.size() << std::endl;
    Mean_vector posterior_mean((int) K(), 0.0);
    Scale_vector posterior_precision((int) K(), 1.0 / prior_precision_);
    for(size_t j = 0; j < J(); ++j)
    {
        kjb::Index_range rows(partition_map(j));
        size_t n_j = rows.size();
        if(n_j > 0)
        {
            kjb::Const_matrix_view Y_j = Y(rows, kjb::Index_range::ALL);
            Mean_vector ybar_j = kjb::sum_matrix_rows(Y_j) / n_j;
            posterior_precision = n_j * h + prior_precision_;
            posterior_mean =
                n_j * ybar_j.ew_multiply(h).ew_divide(posterior_precision);
        }
        for(size_t k = 0; k < K(); ++k)
        {
            Normal_dist r_x_jk(posterior_mean(k), 1.0 / posterior_precision(k));
            X_(j,k) = kjb::sample(r_x_jk);
        }
    }
    std::cerr << "done." << std::endl;
}
 */

void Normal_mean_prior::set_up_results_log() const
{
    create_directory_if_nonexistent(write_path + "X");
}

void Normal_mean_prior::write_state_to_file(const std::string& name) const
{
    PMWP(verbose_ > 0, "    Writing means to file %s/X/%s.txt\n", (write_path.c_str())(name.c_str()));
    //std::cerr << "    Writing means to file " << write_path
    //          << "/X/" << name << ".txt" << std::endl;
    X_.write((write_path + "X/" + name + ".txt").c_str());
}

