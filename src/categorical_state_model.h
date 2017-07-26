/* $Id: categorical_state_model.h 21103 2017-01-22 16:27:51Z chuang $ */

#ifndef CATEGORICAL_STATE_MODEL_H_
#define CATEGORICAL_STATE_MODEL_H_

/*!
 * @file categorical_state_model.h
 *
 * @author Colin Dawson 
 */

#include "parameters.h"
#include "util.h"
#include "state_model.h"
#include <l_cpp/l_int_matrix.h>
#include <boost/shared_ptr.hpp>

class Emission_model;
class Categorical_state_model;
struct Categorical_state_parameters;

typedef boost::shared_ptr<Categorical_state_parameters> Categorical_state_param_ptr;
typedef boost::shared_ptr<Categorical_state_model> Categorical_model_ptr;

struct Categorical_state_parameters : public State_parameters
{
    const bool fixed_alpha;
    const bool identical_alpha_priors;
    const double alpha;
    const double a_alpha;
    const double b_alpha;
    const std::string alpha_file;
    const std::string alpha_prior_file;
    
    Categorical_state_parameters(
        const Parameters & params,
        const std::string& name = "Categorical_state_model"
        );

    static const std::string& bad_alpha_prior()
    {
        static const std::string msg =
            "ERROR: For a Categorical state model, config file must either specify "
            "alpha, specify positive a_alpha and b_alpha, or specify an alpha_prior_file "
            "parameter as the name of a file containing a_alpha and b_alpha for each "
            "latent dimension.";
        return msg;
    }
        
    virtual ~Categorical_state_parameters() {}

    virtual State_model_ptr make_module() const;
};

class Categorical_state_model : public State_model
{
public:
    /*------------------------------------------------------------
     * TYPEDEFS
     *------------------------------------------------------------*/
    typedef Categorical_state_model           Self;
    typedef State_model                       Base_class;
    typedef kjb::Int_vector                   State_count_v;
    typedef std::vector<Prob_vector>          State_count_vv;
    using Base_class::theta;
public:
    /*------------------------------------------------------------
     * CONSTRUCTORS
     *------------------------------------------------------------*/
    Categorical_state_model(
        const Categorical_state_parameters* const  hyperparams
        );

    /*------------------------------------------------------------
     * DESTRUCTOR
     *------------------------------------------------------------*/
    virtual ~Categorical_state_model() {}
    
    /*------------------------------------------------------------
     * INITIALIZERS
     *------------------------------------------------------------*/
    virtual void initialize_params();
    virtual void initialize_resources();
    
    virtual void input_previous_results(const std::string& input_path, const std::string& name);

    /*------------------------------------------------------------
     * DISPLAY FUNCTIONS
     *------------------------------------------------------------*/
    virtual void set_up_results_log() const;
    virtual void write_state_to_file(const std::string& name) const;
    
    /*------------------------------------------------------------
     * INFERENCE INTERFACE
     *------------------------------------------------------------*/
    virtual void update_params();

    /*------------------------------------------------------------
     * EVALUATION INTERFACE
     *------------------------------------------------------------*/
    /*
    virtual kjb::Matrix build_augmented_mean_matrix() const;
     */
    virtual kjb::Matrix build_augmented_mean_matrix(const size_t& i) const;
    virtual void add_ground_truth_eval_header() const;
    
    virtual void compare_state_sequence_to_ground_truth(
        const std::string& ground_truth_path,
        const std::string& name
        ) const;

    /*------------------------------------------------------------
     * ACCESSORS VISIBLE TO OTHER MODULES
     *------------------------------------------------------------*/

    virtual size_t dimension_map(const size_t& dprime) const
    {
        //TODO: make this more efficient
        for(size_t d = 0; d < D(); ++d)
        {
            if(dprime < cum_feature_dimensions_[d+1])
                return d;
        }
        KJB_THROW_2(kjb::Illegal_argument,
                    "Attempted to retrieve latent dimension for a dummy variable"
                    " that does not exist");
    }

    virtual size_t first_theta_prime_col_for_theta_d(const size_t& d)
    {
        return cum_feature_dimensions_[d];
    }
    virtual size_t num_theta_prime_cols_for_theta_d(const size_t& d)
    {
        return feature_dimensions_[d];
    }
    
protected:
    /*------------------------------------------------------------
     * ACCESS TO OTHER MODULES
     *------------------------------------------------------------*/
    int get_theta(const size_t& j, const size_t& d) const
    {
        return theta_(j,d);
    }
    /*------------------------------------------------------------
     * INTERNAL ACCESSORS
     *------------------------------------------------------------*/
    Prob alpha(const size_t& d) const {return alpha_.at(d);}
    Prob_vector entity_counts(const size_t& d) const
    {
        return entity_counts_.at(d);
    }
    Prob entity_counts(const size_t& d, const size_t& s) const
    {
        return entity_counts_.at(d).at(s);
    }

    /*------------------------------------------------------------
     * LVALUE ACCESSORS
     *------------------------------------------------------------*/
    Prob& alpha(const size_t& d) {return alpha_.at(d);}
    Prob& entity_counts(const size_t& d, const size_t& s)
    {
        return entity_counts_.at(d).at(s);
    }
    
    /*------------------------------------------------------------
     * INTERNAL INFERENCE FUNCTIONS
     *------------------------------------------------------------*/
    void update_alpha_();
    void update_theta_(const size_t& j, const size_t& d);
    void update_theta_();
    Prob_vector get_theta_crp_prior(const size_t& d, const size_t& current_s);
    Prob_vector get_theta_transition_log_likelihood(
        const size_t& j,
        const size_t& d
        );
    void sync_cumulative_feature_dimensions();

    /*------------------------------------------------------------
     * HYPERPARAMETERS
     *------------------------------------------------------------*/
    const std::string alpha_file;
    const std::string alpha_prior_file;
    const Scale_vector a_alpha_;
    const Scale_vector b_alpha_;
    const bool fixed_alpha;

    /*------------------------------------------------------------
     * STATE VARIABLES
     *------------------------------------------------------------*/
    Prob_vector alpha_;
    State_matrix dummy_theta_;
    
    /*------------------------------------------------------------
     * CACHE VARIABLES
     *------------------------------------------------------------*/
    State_count_vv entity_counts_;
    Count_vector feature_dimensions_;
    Count_vector cum_feature_dimensions_;
};

#endif
