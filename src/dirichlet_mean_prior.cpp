/* $Id: dirichlet_mean_prior.cpp 21443 2017-06-28 18:13:50Z chuang $ */

/*!
 * @file dirichlet_mean_prior.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "dirichlet_mean_prior.h"
#include "mean_emission_model.h"
#include <boost/make_shared.hpp>

Mean_prior_ptr Dirichlet_mean_prior_parameters::make_module() const
{
    return boost::make_shared<Dirichlet_mean_prior>(this);
}

Dirichlet_mean_prior::Dirichlet_mean_prior(const Params* const hyperparameters)
    : Base_class(hyperparameters),
      symmetric_prior(hyperparameters->symmetric_prior),
      prior_mean_file_(hyperparameters->prior_mean_filename),
      alpha_(hyperparameters->alpha)
{}

void Dirichlet_mean_prior::initialize_resources()
{
    Base_class::initialize_resources();
}

void Dirichlet_mean_prior::initialize_params()
{
    if(symmetric_prior) prior_mean_ = Prob_vector((int) K(), 1.0 / K());
    else prior_mean_ = Prob_vector(prior_mean_file_);
    Prob_matrix unnormed_means((int) J(), (int) K());
    for(size_t k = 0; k < K(); ++k)
    {
        Gamma_dist r_x(alpha_ * prior_mean_.at(k), 1.0);
        for(size_t j = 0; j < J(); ++j)
        {
            unnormed_means(j,k) = log(kjb::sample(r_x));
        }
    }
    X_ = kjb::log_normalize_rows(unnormed_means);
}

void Dirichlet_mean_prior::input_previous_results(const std::string& input_path, const std::string& name)
{
    PMWP(verbose_ > 0, "Inputting information from iteration %s for Dirichlet Mean Prior...\n", (name.c_str()));
    //std::cerr << "Inputting information from iteration " << name << " for Dirichlet Mean Prior..." << std::endl;
    PM(verbose_ > 0, "Inputting information for mena matrix X_...\n");
    //std::cerr << "Inputting information for mean matrix X_..." << std::endl;
    X_ = kjb::ew_log(Mean_matrix((input_path + "X/" + name + ".txt").c_str()));
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
}

void Dirichlet_mean_prior::generate_data(const std::string&)
{
    initialize_params();
    PMWP(verbose_ > 0, "    Writing means to file %smeans.txt\n", (write_path.c_str()));
    //std::cerr << "    Writing means to file " << write_path + "means.txt"
    //          << std::endl;
    kjb::ew_exponentiate(X_).write((write_path + "means.txt").c_str());
};

void Dirichlet_mean_prior::update_params()
{
    PM(verbose_ > 0, "    Updating emission distributions...");
    //std::cerr << "    Updating emission distributions...";
    const Noisy_data_matrix_list& Y = noisy_data();
    Prob_matrix unnormed_means((int) J(), (int) K());
    for(size_t j = 0; j < J(); ++j)
    {
        Count_vector counts_j(K(), 0);
        for (size_t i = 0; i < NF(); i++)
        {
            Time_set times = partition_map(i, j);
            for (Time_set::const_iterator it = times.begin();
                 it != times.end(); ++it)
            {
                counts_j[Y[i](*it, 0)]++;
            }
        }
        for(size_t k = 0; k < K(); ++k)
        {
            Gamma_dist r_x(alpha_ * prior_mean_[k] + counts_j[k], 1.0);
            unnormed_means(j, k) = log(kjb::sample(r_x));
        }
    }
    X_ = kjb::log_normalize_rows(unnormed_means);
}

/*
void Dirichlet_mean_prior::update_params()
{
    std::cerr << "    Updating emission distributions...";
    const Noisy_data_matrix& Y = noisy_data();
    Prob_matrix unnormed_means((int) J(), (int) K());
    for(size_t j = 0; j < J(); ++j)
    {
        Time_set times = partition_map(j);
        Count_vector counts_j(K(), 0);
        for (Time_set::const_iterator it = times.begin();
             it != times.end(); ++it)
        {
            counts_j[Y(*it, 0)]++;
        }
        for(size_t k = 0; k < K(); ++k)
        {
            Gamma_dist r_x(alpha_ * prior_mean_[k] + counts_j[k], 1.0);
            unnormed_means(j, k) = log(kjb::sample(r_x));
        }
    }
    X_ = kjb::log_normalize_rows(unnormed_means);
}
 */

void Dirichlet_mean_prior::set_up_results_log() const
{
    create_directory_if_nonexistent(write_path + "X");
}

void Dirichlet_mean_prior::write_state_to_file(const std::string& name) const
{
    PMWP(verbose_ > 0, "     Writing emission distributions to file %s/X/%s.txt\n",
        (write_path.c_str())(name.c_str()));
    //std::cerr << "    Writing emission distributions to file " << write_path
    //          << "/X/" << name << ".txt" << std::endl;
    kjb::ew_exponentiate(X_).write((write_path + "X/" + name + ".txt").c_str());
}
