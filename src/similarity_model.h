/* $Id: similarity_model.h 21479 2017-07-17 14:08:40Z chuang $ */

#ifndef SIMILARITY_MODEL_H_
#define SIMILARITY_MODEL_H_

/*!
 * @file similarity_model.h
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "transition_prior.h"
#include <boost/shared_ptr.hpp>

class HDP_HMM_LT;
class State_model;
class Similarity_model;
struct Similarity_model_parameters;
typedef boost::shared_ptr<Similarity_model> Similarity_model_ptr;
typedef boost::shared_ptr<Similarity_model_parameters> Similarity_param_ptr;

struct Similarity_model_parameters
{
    Similarity_model_parameters() {}

    virtual ~Similarity_model_parameters() {}

    virtual Similarity_model_ptr make_module() const = 0;
};

class Similarity_model
{
public:
    class HMC_interface;
    typedef boost::shared_ptr<HMC_interface> Model_ptr;
public:
    /*------------------------------------------------------------
     * CONSTRUCTORS
     *------------------------------------------------------------*/
    Similarity_model() : params() {};

    /*------------------------------------------------------------
     * DESTRUCTOR
     *------------------------------------------------------------*/
    virtual ~Similarity_model() {}

    /**
     * @brief set the parent model
     */
    void set_parent(const HDP_HMM_LT* const p);
    
    /*------------------------------------------------------------
     * INITIALIZERS
     *------------------------------------------------------------*/
    virtual void initialize_resources() = 0;
    
    virtual void initialize_params() = 0;
    
    virtual void input_previous_results(const std::string& input_path, const std::string& name) = 0;

    /*------------------------------------------------------------
     * DISPLAY FUNCTIONS
     *------------------------------------------------------------*/
    virtual void set_up_results_log() const;
    virtual void write_state_to_file(const std::string& name) const;
    
    /*------------------------------------------------------------
     * VERBOSE SET UP FUNCTIONS
     *------------------------------------------------------------*/
    void set_up_verbose_level(const size_t verbose){verbose_ = verbose;};

    /*------------------------------------------------------------
     * INFERENCE INTERFACE
     *------------------------------------------------------------*/
    virtual void update_params() = 0;

    /*------------------------------------------------------------
     * ACCESSORS VISIBLE TO OTHER MODULES
     *------------------------------------------------------------*/
    virtual const Prob_matrix& Phi() const {return params->get_Phi();}

    virtual const Prob& Phi(const size_t& j, const size_t& jp) const
    {
        return params->get_Phi(j,jp);
    }

    class HMC_interface
    {
    public:
        friend class Similarity_model;
    public:
        HMC_interface(const Similarity_model* const sim_m);

        Prob log_likelihood() const;
        Prob log_likelihood_gradient(const size_t& j, const size_t& d) const;

        Coordinate get(size_t pos) const;
        void set(size_t pos, Coordinate val);
        size_t size() const;
        
        void initialize_params(const State_matrix& new_theta);

        void sync_Delta();
        void sync_Delta_jd(
            const size_t&     j,
            const size_t&     d,
            const Coordinate& theta_old,
            const Coordinate& theta_new);
        void sync_Phi();
        void sync_anti_Phi();
        void sync_after_theta_update(
            const size_t&     j,
            const size_t&     d,
            const Coordinate& theta_old,
            const Coordinate& theta_new);

        const State_matrix& get_theta() const {return theta;}
        const Prob& get_Phi(const size_t& j, const size_t& jp) const;
        const Prob_matrix& get_Phi() const {return Phi;}
        const Prob_matrix& get_anti_Phi() const {return anti_Phi;}
        const Distance_matrix& get_Delta() const {return Delta;}
        Count N(const size_t& j, const size_t& jp) const;
        Count Q(const size_t& j, const size_t& jp) const;
        
    private:
        const Similarity_model* sim_m;
        size_t J;
        size_t D;
        State_matrix theta;
        Distance_matrix Delta;
        Prob_matrix Phi;
        Prob_matrix anti_Phi;
    };

    Model_ptr get_model() const {return params;}
    
    /*------------------------------------------------------------
     * INFERENCE FUNCTIONS ACCESSIBLE TO OTHER MODULES
     *------------------------------------------------------------*/
    virtual void sync_after_theta_update(
        const size_t&,
        const size_t&,
        const Coordinate&,
        const Coordinate&
        ) const = 0;

    virtual Distance get_change_to_Delta(
        const size_t&       j,
        const size_t&       d,
        const Coordinate&   theta_old,
        const Coordinate&   theta_new,
        const State_matrix& theta
        ) const = 0;

    /*
    virtual Distance get_Phi_j_jp(
        const Distance& Delta_j_jp
        ) const = 0;
     */
    
    virtual Distance get_Phi_j_jp(
        const double&   lambda,
        const Distance& Delta_j_jp
        ) const = 0;
    
    virtual void sync_before_theta_update() const = 0;

    virtual Prob log_likelihood_for_state(
        const size_t&,
        const size_t&,
        const double&
        ) const {return 0.0;}
    
    virtual Prob_vector log_likelihood_for_state_range(
        const size_t& j,
        const size_t& d,
        const size_t& num_states
        ) const;

    /*
    virtual Prob log_likelihood_gradient_term(
        const Coordinate& theta1,
        const Coordinate& theta2,
        const Count&      successes,
        const Count&      failures,
        const Prob&       log_similarity,
        const Prob&       log_dissimilarity
        ) const = 0;
     */
    
    virtual Prob log_likelihood_gradient_term(
        const Coordinate& theta1,
        const Coordinate& theta2,
        const Distance&   delta,
        const Count&      successes,
        const Count&      failures,
        const Prob&       log_similarity,
        const Prob&       log_dissimilarity
        ) const = 0;

    virtual Distance_matrix get_Delta(const State_matrix& theta) const = 0;
    virtual Prob_matrix get_Phi(const double& lambda, const Distance_matrix& delta) const = 0;
    /*
    virtual Prob_matrix get_Phi(const Distance_matrix& delta) const = 0;
     */
    virtual Prob_matrix get_anti_Phi(const double& lambda, const Distance_matrix& Delta) const = 0;
    virtual Prob get_anti_Phi_j_jp(const double& lambda, const Distance& Delta_j_jp) const = 0;
    virtual double lambda() const = 0;

    virtual Model_ptr new_model_object() const = 0;

    /*------------------------------------------------------------
     * ACCESS TO OTHER MODULES
     *------------------------------------------------------------*/
    const size_t& J() const;
    const size_t& D() const;
    const State_matrix& theta() const;
    Coordinate get_theta(const size_t& j, const size_t& d) const;
    State_type theta_prime(const size_t& j) const;
    const Count_matrix& N() const;
    const Count_matrix& Q() const;
    Count N(const size_t& j, const size_t& jp) const;
    Count Q(const size_t& j, const size_t& jp) const;
    
    const size_t& D_prime() const;
protected:
    /*------------------------------------------------------------
     * LVALUE ACCESSORS
     *------------------------------------------------------------*/
    
    virtual Prob& Phi(const size_t& j, const size_t& jp)
    {
        return params->Phi.at(j,jp);
    }
    
    Distance& Delta(const size_t& j, const size_t& jp) const {return params->Delta.at(j,jp);}
    
public:
    std::string write_path;
    
    /*------------------------------------------------------------
     * VERBOSE VARIABLE
     *------------------------------------------------------------*/
    size_t verbose_;
    
protected:
    /*------------------------------------------------------------
     * PARAMETER OBJECT
     *------------------------------------------------------------*/

    Model_ptr params;
    
    /*------------------------------------------------------------
     * LINK VARIABLES
     *------------------------------------------------------------*/
    const HDP_HMM_LT* parent;

    /*------------------------------------------------------------
     * CACHE VARIABLES
     *------------------------------------------------------------*/

    friend class HDP_HMM_LT;
};

#endif
