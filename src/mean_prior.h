/* $Id: mean_prior.h 21443 2017-06-28 18:13:50Z chuang $ */

#ifndef MEAN_PRIOR_H_
#define MEAN_PRIOR_H_

/*!
 * @file mean_prior.h
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "mean_emission_model.h"
#include <boost/shared_ptr.hpp>

class Mean_prior;
struct Mean_prior_parameters;
class Mean_emission_model;

struct Mean_prior_parameters
{
    Mean_prior_parameters() {}

    virtual Mean_prior_ptr make_module() const = 0;
    virtual ~Mean_prior_parameters() {}
};

class Mean_prior
{
public:
    typedef Mean_prior Self;
    typedef Emission_model Parent;
    typedef Mean_prior_parameters Params;
    typedef Parent::Noisy_data_matrix Noisy_data_matrix;
    typedef std::vector<Noisy_data_matrix> Noisy_data_matrix_list;
    typedef Parent::Noise_parameters Noise_parameters;

    /*------------------------------------------------------------
     * CONSTRUCTOR
     *------------------------------------------------------------*/

    Mean_prior(const Params* const) {}
    virtual ~Mean_prior() {}

    /*------------------------------------------------------------
     * INITIALIZERS
     *------------------------------------------------------------*/
    virtual void initialize_resources();
    virtual void initialize_params() = 0;
    void set_parent(Mean_emission_model* const p);
    virtual void generate_data(const std::string& filename) = 0;
    
    virtual void input_previous_results(const std::string& input_path, const std::string& name) = 0;
    
    /*------------------------------------------------------------
     * VERBOSE SET UP FUNCTIONS
     *------------------------------------------------------------*/
    void set_up_verbose_level(const size_t verbose){verbose_ = verbose;};

    /*------------------------------------------------------------
     * INFERENCE INTERFACE
     *------------------------------------------------------------*/
    virtual void update_params() = 0;

    /*------------------------------------------------------------
     * DISPLAY FUNCTIONS
     *------------------------------------------------------------*/
    virtual void set_up_results_log() const = 0;
    virtual void write_state_to_file(const std::string& name) const = 0;

    /*------------------------------------------------------------
     * ACCESSORS NEEDED BY OTHER MODULES
     *------------------------------------------------------------*/

    const Mean_matrix& X() {return X_;}

protected:
    /*------------------------------------------------------------
     * ACCESS TO OTHER MODULES
     *------------------------------------------------------------*/

    size_t J() const;
    size_t K() const;
    size_t T(const size_t& i) const;
    size_t NF() const;
    //why should we need T(i) here?
    /*
    size_t T() const;
     */
    /*
    Noisy_data_matrix noisy_data();
     */
    Noisy_data_matrix_list noisy_data();
    
    Noisy_data_matrix noisy_data (const size_t& i);

    Noise_parameters noise_parameters();

    const Time_set& partition_map(const size_t i, const State_indicator& j) const;

public:
    std::string write_path;

protected:
    /*------------------------------------------------------------
     * LINK VARIABLES
     *------------------------------------------------------------*/
    Mean_emission_model* parent;

    /*------------------------------------------------------------
     * PARAMETERS
     *------------------------------------------------------------*/
    
    Mean_matrix X_;
    
    /*------------------------------------------------------------
     * VERBOSE VARIABLE
     *------------------------------------------------------------*/
    size_t verbose_;
};

#endif
