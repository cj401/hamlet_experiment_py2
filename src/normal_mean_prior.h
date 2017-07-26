/* $Id: normal_mean_prior.h 21103 2017-01-22 16:27:51Z chuang $ */

#ifndef NORMAL_MEAN_PRIOR_H_
#define NORMAL_MEAN_PRIOR_H_

/*!
 * @file normal_mean_prior.h
 *
 * @author Colin Dawson 
 */

#include "parameters.h"
#include "mean_prior.h"

class Normal_mean_prior;
struct Normal_mean_prior_params;
class Mean_emission_model;

struct Normal_mean_prior_params : public Mean_prior_parameters
{
    const double prior_precision;
    
    Normal_mean_prior_params(
        const Parameters&  params,
        const std::string& name = "Normal_mean_prior"
        ) : prior_precision(
                params.get_param_as<double>(name, "precision"))
    {}
            
    virtual Mean_prior_ptr make_module() const;
    virtual ~Normal_mean_prior_params() {}
};

class Normal_mean_prior : public Mean_prior
{
public:
    typedef Normal_mean_prior Self;
    typedef Normal_mean_prior_params Params;
    typedef Mean_prior Base_class;

    /*------------------------------------------------------------
     * CONSTRUCTOR
     *------------------------------------------------------------*/

    Normal_mean_prior(const Params* const hyperparameters);
    
    virtual ~Normal_mean_prior() {}

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

    const double prior_precision_;
};

#endif

