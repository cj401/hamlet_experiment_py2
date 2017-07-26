/* $Id: isotropic_exponential_similarity.h 21479 2017-07-17 14:08:40Z chuang $ */

#ifndef ISOTROPIC_EXPONENTIAL_SIMILARITY_H_
#define ISOTROPIC_EXPONENTIAL_SIMILARITY_H_

/*!
 * @file isotropic_exponential_similarity.h
 *
 * @author Colin Dawson 
 */

#include "parameters.h"
#include "util.h"
#include "similarity_model.h"

class Isotropic_exponential_similarity;
struct Isotropic_exponential_similarity_parameters;

struct Isotropic_exponential_similarity_parameters : public Similarity_model_parameters
{
    const bool fixed_lambda;
    const double lambda;
    const double b_lambda;
    const std::string metric;
    const std::string kernel;

    Isotropic_exponential_similarity_parameters(
        const Parameters&   params,
        const std::string&  name = "Isotropic_exponential_similarity"
        ) : Similarity_model_parameters(),
            fixed_lambda(params.exists(name, "lambda")),
            lambda(
                !fixed_lambda ? NULL_RATE :
                params.get_param_as<double>(
                   name, "lambda", bad_lambda_prior(),
                    valid_decay_param)),
            b_lambda(
                fixed_lambda ? NULL_RATE :
                params.get_param_as<double>(
                    name, "b_lambda", bad_lambda_prior(),
                    valid_rate_param)),
            metric(params.exists(name, "metric") ?
                   params.get_param_as<std::string>(name, "metric") :
                   "hamming"),
            kernel(params.exists(name, "kernel") ?
                   params.get_param_as<std::string>(name, "kernel") :
                   "isotropic_exponential")
    {}

    Isotropic_exponential_similarity_parameters()
        : fixed_lambda(true),
          lambda(0.0),
          b_lambda(NULL_RATE),
          metric("hamming"),
          kernel("isotropic_exponential")
    {}
        
    virtual ~Isotropic_exponential_similarity_parameters() {}
    
    static const std::string& bad_lambda_prior()
    {
        static const std::string msg =
            "ERROR: For an exponential similarity kernel, "
            "config file must specify a valid positive lambda or b_lambda.";
        return msg;
    }

    virtual Similarity_model_ptr make_module() const;
                         
};

class Isotropic_exponential_similarity : public Similarity_model
{
public:
    typedef Isotropic_exponential_similarity Self;
    typedef Similarity_model Base_class;
    typedef Isotropic_exponential_similarity_parameters Params;
    using Base_class::Model_ptr;
public:
    /*------------------------------------------------------------
     * CONSTRUCTORS
     *------------------------------------------------------------*/
    Isotropic_exponential_similarity(
        const Params* const hyperparams
        );

    /*------------------------------------------------------------
     * DESTRUCTOR
     *------------------------------------------------------------*/
    virtual ~Isotropic_exponential_similarity() {}

    /*------------------------------------------------------------
     * INITIALIZERS
     *------------------------------------------------------------*/
    virtual void initialize_resources();
    virtual void initialize_params();

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
     * INFERENCE FUNCTIONS ACCESSIBLE TO OTHER MODULES
     *------------------------------------------------------------*/
    virtual void sync_after_theta_update(
        const size_t&     j,
        const size_t&     d,
        const Coordinate& theta_old,
        const Coordinate& theta_new
        ) const;
    
    virtual void sync_before_theta_update() const;
    
    virtual Prob log_likelihood_for_state(
        const size_t& j,
        const size_t& d,
        const double& theta_jd
        ) const;

    /*
    Prob log_likelihood_gradient_term(
        const Coordinate& pos_1,
        const Coordinate& pos_2,
        const Count&      successes,
        const Count&      failures,
        const Prob&       log_similarity,
        const Prob&       log_dissimilarity
        ) const;
     */
    Prob log_likelihood_gradient_term(
        const Coordinate& pos_1,
        const Coordinate& pos_2,
        const Distance&   delta,
        const Count&      successes,
        const Count&      failures,
        const Prob&       log_similarity,
        const Prob&       log_dissimilarity
        ) const;

    Model_ptr new_model_object() const;

protected:
    
    /*------------------------------------------------------------
     * INTERNAL ACCESSORS
     *------------------------------------------------------------*/
    const Count_matrix& Q_part() {return Q_part_;}
    const Count_matrix& N_part() {return N_part_;}

    /*------------------------------------------------------------
     * LVALUE ACCESSORS
     *------------------------------------------------------------*/
    /*
    Count& C(const size_t& j, const size_t& d, const int& theta) const
    {
        if(C_.find(theta) == C_.end())
        {
            C_[theta] = Count_matrix(J(),D(),0);
        }
        return C_.at(theta).at(j,d);
    }
     */
    Count& C(const size_t& j, const size_t& d, const int& theta) const
    {
        if(C_.find(theta) == C_.end())
        {
            C_[theta] = Count_matrix(J(),D_prime(),0);
        }
        return C_.at(theta).at(j,d);
    }

    /*------------------------------------------------------------
     * INTERNAL INFERENCE FUNCTIONS
     *------------------------------------------------------------*/
    void sync_C_() const;
    
    Distance_matrix get_Delta(const State_matrix& theta) const;

    Distance get_change_to_Delta(
        const size_t&       j,
        const size_t&       d,
        const Coordinate&   theta_old,
        const Coordinate&   theta_new,
        const State_matrix& theta
        ) const;
    
    /*
    Prob get_Phi_j_jp(const Distance& Delta_j_jp) const;

    Prob_matrix get_Phi(const Distance_matrix& Delta) const;
     */
    
    Prob get_Phi_j_jp(const double& lambda, const Distance& Delta_j_jp) const;
    
    Prob_matrix get_Phi(const double& lambda, const Distance_matrix& Delta) const;
    
    void sync_C_jd_(
        const size_t& j,
        const size_t& d,
        const Coordinate& theta_old,
        const Coordinate& theta_new
        ) const;
    void update_lambda_();
    Prob_matrix get_anti_Phi(const double& lambda, const Distance_matrix& Delta) const;
    Prob get_anti_Phi_j_jp(const double& lambda, const Distance& Delta_j_jp) const;
    double lambda() const {return lambda_;}
    friend double lambda_log_density(double lambda, Self* model);

protected:
    
    /*------------------------------------------------------------
     * HYPERPARAMETERS
     *------------------------------------------------------------*/
    const bool fixed_lambda;
    const double b_lambda_;
    double (*metric_)(const double&, const double&);
    const std::string kernel_;

    /*------------------------------------------------------------
     * STATE VARIABLES
     *------------------------------------------------------------*/
    double lambda_;
    
    /*------------------------------------------------------------
     * CACHE VARIABLES
     *------------------------------------------------------------*/    
    mutable Count_matrix Q_part_;
    mutable Count_matrix N_part_;
    mutable std::map<size_t, Count_matrix> C_;
    //mutable double lambda_posterior_linear_coef_;
};

#endif

