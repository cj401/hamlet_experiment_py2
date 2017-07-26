/* $Id: continuous_state_model.cpp 21468 2017-07-11 19:35:05Z cdawson $ */

/*!
 * @file continuous_state_model.cpp
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "continuous_state_model.h"
#include "similarity_model.h"
#include <boost/make_shared.hpp>
#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>
#include <ergo/hmc.h>

class HMC_adaptor;

Continuous_state_parameters::Continuous_state_parameters(
    const Parameters&  params,
    const std::string& name
    ) : State_parameters(params, name),
        prior_precision(
            params.get_param_as<double>(
                name, "prior_precision", bad_prior_precision(), valid_shape_param)),
        L(params.get_param_as<size_t>(name, "L")),
        epsilon(params.get_param_as<double>(name, "epsilon"))
{}

State_model_ptr Continuous_state_parameters::make_module() const
{
    return boost::make_shared<Continuous_state_model>(this);
}

Continuous_state_model::Continuous_state_model(
    const Continuous_state_parameters* const hyperparams
    ) : State_model(hyperparams),
        prior_precision_(hyperparams->prior_precision),
        L_(hyperparams->L),
        epsilon_(hyperparams->epsilon),
        last_accepted(0),
        acceptance_rate(0.0),
        starting_iterations(0),
        starting_num_accepted(0)
{}

void Continuous_state_model::initialize_params()
{
    PM(verbose_ > 0, "Initializing continuous state model...\n");
    PM(verbose_ > 0, "     Initializing theta...");
    //std::cerr << "Initializing continuous state model..." << std::endl;
    //std::cerr << "    Initializing theta...";
    for(size_t j = 0; j < J(); ++j)
    {
        for(size_t d = 0; d < D(); ++d)
        {
            Normal_dist r_theta(0.0, 1.0 / prior_precision_);
            theta(j,d) = kjb::sample(r_theta);
        }
    }
    PM(verbose_ > 0, "done.\n");
    //std::cerr << "done." << std::endl;
    theta_prime_ = &theta_;
}

void Continuous_state_model::initialize_resources()
{
    PM(verbose_ > 0, "Initializing resources for continuous state model...\n");
    //std::cerr << "Initializing resources for continuous state model..."
    //          << std::endl;
    Base_class::initialize_resources();
}

void Continuous_state_model::input_previous_results(const std::string& input_path, const std::string& name)
{
    PM(verbose_ > 0, "Inputting information for continuous state model...\n");
    //std::cerr << "Inputting information for continuous state model..." << std::endl;
    Base_class::input_previous_results(input_path, name);
    if (input_path == write_path)
    {
        starting_iterations = boost::lexical_cast<size_t>(name);
        acceptance_rate = input_to_value<double>(input_path, "hmc_acceptance_rate.txt", name);
        PMWP(verbose_ > 0, "acceptance_rate_ = %.6f\n", (acceptance_rate));
        //std::cerr << "acceptance_rate_ = " << acceptance_rate << std::endl;
        //starting_num_accepted = boost::lexical_cast<size_t>(starting_iterations * acceptance_rate);
        starting_num_accepted = (int)(starting_iterations * acceptance_rate);
        PMWP(verbose_ > 0, "starting_num_accepted_ = %d\n", (starting_num_accepted));
        //std::cerr << "starting_num_accepted_ = " << starting_num_accepted << std::endl;
    }
}

void Continuous_state_model::set_up_results_log() const
{
    Base_class::set_up_results_log();
    std::ofstream ofs;
    ofs.open(write_path + "hmc_acceptances.txt", std::ofstream::out);
    ofs << "iteration value" << std::endl;
    ofs.close();
    ofs.open(write_path + "hmc_acceptance_rate.txt", std::ofstream::out);
    ofs << "iteration value" << std::endl;
    ofs.close();
}

void Continuous_state_model::write_state_to_file(const std :: string & name) const
{
    const std::string filestem = write_path + name;
    std::ofstream ofs;
    kjb::Matrix(theta()).submatrix(0,0,J(), D()).write(
        (write_path + "theta/" + name + ".txt").c_str());
    ofs.open(write_path + "hmc_acceptance_rate.txt", std::ofstream::app);
    ofs << name << " " << acceptance_rate << std::endl;
    ofs.close();
}

void Continuous_state_model::update_params()
{
    PM(verbose_ > 0, "Updating continuous state parameters...\n");
    //std::cerr << "Updating continuous state parameters..." << std::endl;
    update_theta_();
}

void Continuous_state_model::add_ground_truth_eval_header() const
{
    //// Doesn't make a lot of sense to do this here.
}

void Continuous_state_model::compare_state_sequence_to_ground_truth(const std :: string &, const std :: string &) const
{
    //// Doesn't make a lot of sense to do this here.
}

class HMC_adaptor
{
public:
    double get(const Similarity_model::HMC_interface* model, size_t pos) const
    {
        return model->get(pos);
    }
    void set(Similarity_model::HMC_interface* model, size_t pos, Coordinate val)
    {
        model->set(pos, val);
    }
    size_t size(const Similarity_model::HMC_interface* model) const
    {
        return model->size();
    }
};

double Continuous_state_model::theta_log_posterior(
    const Similarity_model::HMC_interface& model) const
{
    return model.log_likelihood() - 0.5 * prior_precision_ *
        kjb::sum_squared_elements(model.get_theta());
}

std::vector<Prob> Continuous_state_model::theta_log_posterior_gradient(
    const Similarity_model::HMC_interface& model) const
{
    std::vector<Prob> result((int) J() * D());
    for(size_t d = 0; d < D(); ++d)
    {
        for(size_t j = 0; j < J(); ++j)
        {
            result[j*D() + d] = model.log_likelihood_gradient(j,d) -
                2 * prior_precision_ * theta(j,d);
        }
    }
    return result;
}

void Continuous_state_model::update_theta_()
{
    static size_t iterations = starting_iterations;
    static size_t num_accepted = starting_num_accepted;
    
    //std::cerr << "num_accepted = " << num_accepted << std::endl;
    
    typedef Similarity_model::HMC_interface Model;

    Model& model = *(similarity_model()->get_model());

    HMC_adaptor adaptor;

    ergo::hmc_step<Model>::vec_t step_sizes(model.size(), epsilon_);

    PMWP(verbose_ > 0, "    Current log likelihood is %.6f\n", (theta_log_posterior(model)));
    //std::cerr << "    Current log likelihood is "
    //          << theta_log_posterior(model)
    //          << std::endl;

    // std::cerr << "    Current log likelihood gradient is "
    //           << theta_log_posterior_gradient(model)
    //           << std::endl;
    ergo::hmc_step<Model> step(
        adaptor,
        boost::bind(&Continuous_state_model::theta_log_posterior, this, _1),
        boost::bind(&Continuous_state_model::theta_log_posterior_gradient, this, _1),
        step_sizes, L_, 0.0);

    double lt_q = theta_log_posterior(model);
    step(model, lt_q);
    theta_ = model.get_theta();
    if(step.accepted())
    {
        last_accepted = 1;
        num_accepted++;
    } else {
        last_accepted = 0;
    }
    iterations++;
    std::ofstream ofs;
    ofs.open(write_path + "hmc_acceptances.txt", std::ofstream::app);
    ofs << iterations << " " << last_accepted << std::endl;
    ofs.close();
    PMWP(verbose_ > 0, "    Acceptance probability = %.6f\n", (exp(step.acceptance_probability())));
    //std::cerr << "    Acceptance probability = " << exp(step.acceptance_probability())
    //          << std::endl;
    PMWP(verbose_ > 0, "    Accepted = %d\n", (step.accepted()));
    //std::cerr << "    Accepted = " << step.accepted() << std::endl;
    acceptance_rate = ((double) num_accepted) / iterations;
    PMWP(verbose_ > 0, "    Cumulative accpetance rate = %.6f\n", (acceptance_rate));
    //std::cerr << "    Cumulative acceptance rate = " << acceptance_rate
    //          << std::endl;
}
