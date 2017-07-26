/* $Id: dirichlet_mean_prior.h 21103 2017-01-22 16:27:51Z chuang $ */

#ifndef DIRICHLET_MEAN_PRIOR_H_
#define DIRICHLET_MEAN_PRIOR_H_

/*!
 * @file dirichlet_mean_prior.h
 *
 * @author Colin Dawson 
 */

#include "mean_emission_model.h"
#include "mean_prior.h"
#include "parameters.h"

class Dirichlet_mean_prior;
struct Dirichlet_mean_prior_parameters;

struct Dirichlet_mean_prior_parameters : public Mean_prior_parameters
{
    typedef Mean_prior_parameters Base_class;
    
    const bool symmetric_prior;
    const double alpha;
    const std::string prior_mean_filename;
    
    Dirichlet_mean_prior_parameters(
        const Parameters&   params,
        const std::string&  name = "Dirichlet_multinomial_emissions"
        ) : Base_class(),
            symmetric_prior(!params.exists(name, "prior_mean_file")),
            alpha(
                params.get_param_as<double>(
                    name, "alpha", bad_alpha_prior(), valid_conc_param)),
            prior_mean_filename(
                params.exists(name, "prior_mean_file") ?
                params.get_param_as<std::string>(name, "prior_mean_file") :
                "")
    {}

    virtual ~Dirichlet_mean_prior_parameters() {};
        
    static const std::string& bad_alpha_prior()
    {
        static const std::string msg = 
            "ERROR: For a Dirichlet mean prior, "
            " config file must specify alpha > 0.";
        return msg;
    }

    virtual Mean_prior_ptr make_module() const;
};

class Dirichlet_mean_prior : public Mean_prior
{
public:
    typedef Dirichlet_mean_prior Self;
    typedef Mean_prior Base_class;
    typedef Dirichlet_mean_prior_parameters Params;

public:
    /*------------------------------------------------------------
     * CONSTRUCTOR
     *------------------------------------------------------------*/

    Dirichlet_mean_prior(const Params* const hyperparameters);

    virtual ~Dirichlet_mean_prior() {}

    /*------------------------------------------------------------
     * INITIALIZERS
     *------------------------------------------------------------*/

    virtual void initialize_resources();
    virtual void initialize_params();
    virtual void generate_data(const std::string& filename);
    
    virtual void input_previous_results(const std::string& input_path, const std::string& name);

    /*------------------------------------------------------------
     * INFERENCE INTERFACE
     *------------------------------------------------------------*/

    virtual void update_params();

    /*------------------------------------------------------------
     * DISPLAY FUNCTIONS
     *------------------------------------------------------------*/

    virtual void set_up_results_log() const;

    virtual void write_state_to_file(const std::string& name) const;

protected:

    /*------------------------------------------------------------
     * HYPERPARAMETERS
     *------------------------------------------------------------*/

    const bool symmetric_prior;
    const std::string prior_mean_file_;
    Prob_vector prior_mean_;
    const double alpha_;
};

#endif

