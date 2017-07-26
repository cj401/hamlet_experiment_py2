/* $Id: weight_prior.h 21443 2017-06-28 18:13:50Z chuang $ */

#ifndef WEIGHT_PRIOR_H_
#define WEIGHT_PRIOR_H_

/*!
 * @file weight_prior.h
 *
 * @author Colin Dawson 
 */

#include <m_cpp/m_matrix.h>
#include <string>
#include <boost/shared_ptr.hpp>
#include "noise_model.h"
#include "emission_model.h"

class Weight_prior;
struct Weight_prior_parameters;
class Linear_emission_model;

typedef boost::shared_ptr<Weight_prior_parameters> Weight_param_ptr;
typedef boost::shared_ptr<Weight_prior> Weight_prior_ptr;

struct Weight_prior_parameters
{
    const bool   include_bias;
    Weight_prior_parameters(
        const Parameters& params,
        const std::string& name = "Weight_prior"
        ) : include_bias(
                params.exists(name, "include_bias") ?
                params.get_param_as<bool>(name, "include_bias") :
                true
            )
    {}

    virtual Weight_prior_ptr make_module() const = 0;
    virtual ~Weight_prior_parameters() {}
};

class Weight_prior
{
public:
    typedef kjb::Vector Weight_vector;
    typedef kjb::Matrix Weight_matrix;
    friend class Linear_emission_model;
    typedef Weight_prior_parameters Params;
    typedef Noise_model::Noise_parameters Noise_parameters;
public:
    /*------------------------------------------------------------
     * CONSTRUCTOR
     *------------------------------------------------------------*/
    Weight_prior(const Params* const hyperparameters)
        : W_(),
          b_(),
          include_bias_(hyperparameters->include_bias),
          parent()
    {}
    
    /*------------------------------------------------------------
     * DESTRUCTOR
     *------------------------------------------------------------*/
    virtual ~Weight_prior() {}
    
    /*------------------------------------------------------------
     * INITIALIZERS
     *------------------------------------------------------------*/
    virtual void initialize_resources();
    virtual void initialize_params() = 0;

    virtual void input_previous_results(const std::string& input_path, const std::string& name) = 0;

    void set_parent(Linear_emission_model* const p);
    virtual void generate_data(const std::string& filename) = 0;
    
    /*------------------------------------------------------------
     * VERBOSE SET UP FUNCTIONS
     *------------------------------------------------------------*/
    void set_up_verbose_level(const size_t verbose){verbose_ = verbose;};

    /*------------------------------------------------------------
     * INFERENCE INTERFACE
     *------------------------------------------------------------*/
    virtual void update_params() = 0;
    virtual void insert_latent_dimension(const size_t& d, const size_t& new_pos);
    virtual void remove_latent_dimension(const size_t& old_pos);
    virtual void replace_latent_dimension(const size_t& d, const size_t& pos);
    virtual Weight_vector propose_weight_vector(const size_t& d) const = 0;

    /*------------------------------------------------------------
     * DISPLAY FUNCTIONS
     *------------------------------------------------------------*/
    virtual void set_up_results_log() const;
    
    virtual void write_state_to_file(const std::string& name) const = 0;

    /*------------------------------------------------------------
     * ACCESSORS VISIBLE TO OTHER MODULES
     *------------------------------------------------------------*/
    const Weight_matrix& W() const {return W_;}

    const Weight_vector& get_bias() const {return b_;}
    
    Mean_matrix get_mean_matrix(const State_matrix& theta_star);
    Mean_vector get_mean_vector(const State_type& theta_j);
    
    bool includes_bias() const {return include_bias_;}
protected:
    /*------------------------------------------------------------
     * ACCESS TO OTHER MODULES
     *------------------------------------------------------------*/
    size_t NF() const;
    size_t D() const;
    size_t D_prime() const;
    size_t K() const;
    Noise_parameters noise_parameters() const;
protected:
    Weight_matrix W_;
    Weight_vector b_; //!< vector of bias terms
    const bool include_bias_; //!< whether the bias terms are used
    Linear_emission_model* parent;
    std::string write_path;
    
    /*------------------------------------------------------------
     * VERBOSE VARIABLE
     *------------------------------------------------------------*/
    size_t verbose_;
};

#endif

