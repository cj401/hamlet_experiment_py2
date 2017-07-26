/* $Id: noise_model.h 21468 2017-07-11 19:35:05Z cdawson $ */

#ifndef NOISE_MODEL_H_
#define NOISE_MODEL_H_

/*!
 * @file noise_model.h
 *
 * @author Colin Dawson 
 */

#include "parameters.h"
#include "util.h"
#include <l_cpp/l_index.h>
#include <m_cpp/m_vector.h>
#include <m_cpp/m_matrix.h>
#include <boost/shared_ptr.hpp>

class Emission_model;

class Noise_model;
struct Noise_model_parameters;

typedef boost::shared_ptr<Noise_model_parameters> Noise_param_ptr;
typedef boost::shared_ptr<Noise_model> Noise_model_ptr;

struct Noise_model_parameters
{
    Noise_model_parameters(
        const Parameters& params,
        const std::string& name = "noise_model"
        )
    {}
    
    virtual ~Noise_model_parameters() {}

    virtual Noise_model_ptr make_module() const = 0;
};

/*
struct Noise_model_parameters
{
    virtual Noise_model_ptr make_module() const = 0;
    virtual ~Noise_model_parameters() {}
};
 */

class Noise_model
{
public:
    typedef Noise_model_parameters Params;
    typedef kjb::Matrix Noisy_data_matrix;
    typedef std::vector<Noisy_data_matrix> Noisy_data_matrix_list;
    typedef kjb::Matrix Noise_parameters;
    friend class Emission_model;
public:
    /*------------------------------------------------------------
     * CONSTRUCTORS
     *------------------------------------------------------------*/
    /*
    Noise_model() : T_(0), K_(0), parent() {}
     */
    Noise_model(
        const Noise_model_parameters* const hyperparameters
        ) : T_(T_list()),
            K_(0), parent()
            {}

    /*------------------------------------------------------------
     * DESTRUCTOR
     *------------------------------------------------------------*/
    virtual ~Noise_model() {}

    /*------------------------------------------------------------
     * INITIALIZATION
     *------------------------------------------------------------*/
    virtual void initialize_resources()
    {
        PM(verbose_ > 0, "    Allocating resources for noise model...");
        PM(verbose_ > 0, "done.");
    };
    virtual void initialize_params() {};
    
    virtual void input_previous_results(const std::string& input_path, const std::string& name) = 0;

    void set_parent(Emission_model* const p);
    
    virtual void add_data(const std::string& filename, const size_t& num_files) = 0;
    virtual void add_test_data(const std::string& filename, const size_t& num_files) = 0;
    virtual void generate_data(const std::string& filename) = 0;
    virtual void generate_test_data(const std::string& filename) = 0;
    
    /*------------------------------------------------------------
     * VERBOSE SET UP FUNCTIONS
     *------------------------------------------------------------*/
    void set_up_verbose_level(const size_t verbose){verbose_ = verbose;};

    /*------------------------------------------------------------
     * INFERENCE INTERFACE
     *------------------------------------------------------------*/
    virtual void update_params()
    {
        std::cerr << "Updating noise parameters..." << std::endl;
    }

    /*------------------------------------------------------------
     * DISPLAY FUNCTIONS
     *------------------------------------------------------------*/
    virtual void set_up_results_log() const {};
    virtual void write_state_to_file(const std::string& name) const = 0;

    /*------------------------------------------------------------
     * ACCESSORS VISIBLE TO OTHER MODULES
     *------------------------------------------------------------*/
    
    virtual Likelihood_matrix get_log_likelihood_matrix(
        const size_t&            i,
        const Mean_matrix&       X,
        const kjb::Index_range&  times = kjb::Index_range::ALL
        ) const
    {
        return get_log_likelihood_matrix(i, X, times, T(i), &Noise_model::log_likelihood);
    };
    
    /*
    virtual Likelihood_matrix get_log_likelihood_matrix(
        const Mean_matrix&       X,
        const kjb::Index_range&  times = kjb::Index_range::ALL
        ) const
    {
        return get_log_likelihood_matrix(X, times, T(), &Noise_model::log_likelihood);
    };
     */

    virtual Likelihood_matrix get_test_log_likelihood_matrix(
        const size_t&           i,
        const Mean_matrix&      X,
        const kjb::Index_range& times = kjb::Index_range::ALL
        ) const
    {
        return get_log_likelihood_matrix(i, X, times, test_T(i),
                                         &Noise_model::test_log_likelihood);
    };

    /**
     * @brief return the TxJ' log likelihood matrix for each of J' possible
     *        changes to the current means.
     *
     * @param X_others the current (or baseline) mean matrix
     * @param delta_x a matrix each row of which represents a possible change to X_others
     *
     * @returns a TxJ' matrix where each row represents a time step and each column represents
     *          a log likelihood for change j' \in {1, ..., J'}
     */
    virtual Likelihood_matrix get_conditional_log_likelihood_matrix(
        const size_t&      i,
        const Mean_matrix& X_others,
        const Mean_matrix& delta_x
        );
    /*
    virtual Likelihood_matrix get_conditional_log_likelihood_matrix(
        const Mean_matrix& X_others,
        const Mean_matrix& delta_x
        );
     */

    virtual Noise_parameters parameters() const = 0;
    
    virtual Noisy_data_matrix_list noisy_data() const = 0;
    
    virtual Noisy_data_matrix noisy_data(const size_t& i) const = 0;
    
    /*
    virtual Noisy_data_matrix noisy_data() const = 0;
     */

    /*------------------------------------------------------------
     * COMPUTATION NEEDED BY OTHER MODULES
     *------------------------------------------------------------*/
    virtual double log_likelihood_ratio_for_state_change(
        const size_t&           i,
        const Mean_vector&      x,
        const Mean_vector&      delta,
        const kjb::Index_range& times
        ) = 0;
    
    virtual Prob_vector log_likelihood_ratios_for_state_changes(
        const size_t&           i,
        const Mean_vector&      x,
        const Mean_matrix&      delta,
        const kjb::Index_range& times
        );
    
protected:
    /*------------------------------------------------------------
     * INTERNAL ACCESSORS
     *------------------------------------------------------------*/
    // Prob llm(const size_t& t, const size_t& j) const {return llm_.at(t,j);}
    
    /*------------------------------------------------------------
     * LVALUE ACCESSORS
     *------------------------------------------------------------*/
    // Prob& llm(const size_t& t, const size_t& j) {return llm_.at(t,j);}
    
    /*------------------------------------------------------------
     * MANIPULATORS
     *------------------------------------------------------------*/
    // void set_data_size(const size_t& T, const size_t& K) {T_ = T; K_ = K;}
    
    void set_data_size(const T_list& T, const size_t& K) {T_ = T; K_ = K;}
    
    void set_test_data_size(const T_list& T) {test_T_ = T;}

    //void set_test_data_size(const size_t& T) {test_T_ = T;}
    
    /*------------------------------------------------------------
     * ACCESS TO OTHER MODULES
     *------------------------------------------------------------*/

    // const Mean_matrix& X() const {return parent->X();}
    
    /*
    const size_t& T() const {return T_;}
     */
    
    size_t NF() const;
    
    size_t test_NF() const;
    
    const T_list& T() const {return T_;};
    
    size_t T(const size_t& i) const {return T_.at(i);}
    
    const T_list& test_T() const {return test_T_;}
    
    size_t test_T(const size_t& i) const {return test_T_.at(i);}

    size_t K() const {return K_;}
    
    // const size_t& J() const {return parent->J();}

    /*------------------------------------------------------------
     * INTERNAL COMPUTATION
     *------------------------------------------------------------*/
    virtual double log_likelihood(const size_t& i, const size_t& t, const Mean_vector& mean) const = 0;
    virtual double test_log_likelihood(const size_t& i, const size_t& t, const Mean_vector& mean) const = 0;
    /*
    virtual double log_likelihood(const size_t& t, const Mean_vector& mean) const = 0;
     virtual double test_log_likelihood(const size_t& t, const Mean_vector& mean) const = 0;
     */

    /*------------------------------------------------------------
     * INTERNAL INFERENCE FUNCTIONS
     *------------------------------------------------------------*/
    
    Prob_matrix get_log_likelihood_matrix(
        const size_t&              i,
        const Mean_matrix&         X,
        const kjb::Index_range&    times,
        const size_t&              all_T,
        Prob (Noise_model::* log_likelihood_f)(const size_t& i, const size_t& t, const Mean_vector& mean) const
        ) const;

    Prob_matrix get_log_likelihood_matrix(
        const Mean_matrix&         X,
        const kjb::Index_range&    times,
        const size_t&              all_T,
        Prob (Noise_model::* log_likelihood_f)(const size_t& t, const Mean_vector& mean) const
        ) const;

    /*------------------------------------------------------------
     * CONSTANTS
     *------------------------------------------------------------*/
    /*
    size_t T_;
    size_t test_T_;
     */
    T_list T_;
    T_list test_T_;
    size_t K_;
    
    /*------------------------------------------------------------
     * VERBOSE VARIABLES
     *------------------------------------------------------------*/
    size_t verbose_;

    /*------------------------------------------------------------
     * LINKS
     *------------------------------------------------------------*/    
    Emission_model* parent;
    std::string write_path;
};

#endif

