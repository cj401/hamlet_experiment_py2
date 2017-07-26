/* $Id: normal_noise_model.h 21245 2017-02-13 18:27:33Z chuang $ */

#ifndef NORMAL_NOISE_MODEL_H_
#define NORMAL_NOISE_MODEL_H_

/*!
 * @file normal_noise_model.h
 *
 * @author Colin Dawson 
 */

#include "parameters.h"
#include "util.h"
#include "dynamics_model.h"
#include "emission_model.h"
#include "state_model.h"
#include "noise_model.h"
#include <l_cpp/l_index.h>
#include <m_cpp/m_vector.h>
#include <m_cpp/m_matrix.h>
#include <m_cpp/m_mat_view.h>

class Normal_noise_model;
class Parameters;

struct Normal_noise_parameters : public Noise_model_parameters
{
    const double a_h; //!< shape hyperparameter on noise variance prior
    const double b_h; //!< rate hyperparameter on noise variance prior
    
    Normal_noise_parameters(
        const Parameters & params,
        const std::string& name = "Normal_noise_model"
        );

    static const std::string& bad_h_prior()
    {
        static const std::string msg =
            "I/O ERROR: For a Normal noise model, config file must either "
            "specify precision matrix H, or a_h and b_h, which must be "
            "positive.";
        return msg;
    }

    virtual ~Normal_noise_parameters() {}

    virtual Noise_model_ptr make_module() const;
};

class Normal_noise_model : public Noise_model
{
public:
    /*------------------------------------------------------------
     * TYPEDEFS
     *------------------------------------------------------------*/
    typedef Normal_noise_model Self;
    typedef Noise_model Base_class;
    typedef kjb::Vector Precision_vector;
    typedef kjb::Vector Data_vector;
    typedef Noise_model::Noisy_data_matrix Data_matrix;
    typedef Noise_model::Noisy_data_matrix_list Data_matrix_list;
    // using Base_class::J;
    using Base_class::T;
public:

    /*------------------------------------------------------------
     * CONSTRUCTORS
     *------------------------------------------------------------*/

    /**
     * @brief construct from a collection of hyperparameters
     */
    /*
    Normal_noise_model(
        const Normal_noise_parameters* const hyperparameters
        ) :
        a_h_(hyperparameters->a_h),
        b_h_(hyperparameters->b_h),
        h_(), Y_()
    {}
     */
    Normal_noise_model(
        const Normal_noise_parameters* const hyperparameters
        ) : Noise_model(hyperparameters),
            a_h_(hyperparameters->a_h),
            b_h_(hyperparameters->b_h),
            h_(), Y_()
            {}
    
public:
    /*------------------------------------------------------------
     * DESTRUCTOR
     *------------------------------------------------------------*/
    /**
     * @brief virtual destructor
     */
    virtual ~Normal_noise_model() {}

    /*------------------------------------------------------------
     * INITIALIZERS
     *------------------------------------------------------------*/

    /**
     * @brief allocate memory for parameter objects
     */
    virtual void initialize_resources();

    /**
     * @brief initialize values of the parameter objects
     */
    virtual void initialize_params();

    /**
     *@brief input from previous results
     */

    virtual void input_previous_results(const std::string& input_path, const std::string& name);
    
    /**
     * @brief read in a dataset from a file
     */
    virtual void add_data(const std::string& filename, const size_t& num_files);

    /**
     * @brief read in a test dataset from a file
     */
    virtual void add_test_data(const std::string& filename, const size_t& num_files);
    
    /**
     * @brief generate a dataset and write to file <path>obs.txt
     *
     */
    virtual void generate_data(const std::string& path);

    /**
     * @brief generate a test dataset of size T and write to file <path>obs.txt
     *
     */
    virtual void generate_test_data(const std::string& path);
    
    /*------------------------------------------------------------
     * DISPLAY FUNCTIONS
     *------------------------------------------------------------*/

    virtual void set_up_results_log() const;
    
    /**
     * @brief record the current state of all present parameters
     * @param filename stem for the output
     * @param iteration a label for the present iteration to append
     *        to the filename
     */
    virtual void write_state_to_file(const std::string& name) const;

    /*------------------------------------------------------------
     * INFERENCE INTERFACE
     *------------------------------------------------------------*/
    /**
     * @brief do a Gibbs update step
     */
    virtual void update_params();

    /*------------------------------------------------------------
     * ACCESSORS NEEDED BY OTHER MODULES
     *------------------------------------------------------------*/
    
    /*
    Data_matrix noisy_data() const {return Y_;}
     */
    
    Data_matrix_list noisy_data() const {return Y_;}
    Data_matrix noisy_data(const size_t& i) const {return Y_.at(i);}
    
    Noise_parameters parameters() const {return Noise_parameters(h_);}

    /*------------------------------------------------------------
     * COMPUTATION NEEDED BY OTHER MODULES
     *------------------------------------------------------------*/

    /**
     * @brief log lr relative to current value if theta_jd is shifted by delta
     */
    double log_likelihood_ratio_for_state_change(
        const size_t&           i,
        const Mean_vector&      x,
        const Mean_vector&      delta,
        const kjb::Index_range& times
        );
    
protected:

    /*------------------------------------------------------------
     * INTERNAL COMPUTATION
     *------------------------------------------------------------*/

    /**
     * @brief sample noise precisions from the prior
     */
    void initialize_h_from_prior_();

    /**
     * @brief return the log likelihood of a particular observation vector
     *        given a particular mean vector
     */
    double log_likelihood(
        const size_t&      i,
        const size_t&      t,
        const Mean_vector& means
        ) const;
    /*
    double log_likelihood(
        const size_t&      t,
        const Mean_vector& means
        ) const;
     */

    /**
     * @brief return the log likelihood of a particular test observation
     *        given a particular mean vector
     */
    double test_log_likelihood(
        const size_t&      i,
        const size_t&      t,
        const Mean_vector& means
        ) const;
    /*
    double test_log_likelihood(
        const size_t&      t,
        const Mean_vector& means
        ) const;
     */
    
    /**
     * @brief return the log likelihood of a particular observation vector
     *        given a particular mean vector and a particular data matrix
     */
    static double log_likelihood(
        const Data_matrix&        data,
        const size_t&             t,
        const Mean_vector&        means,
        const Precision_vector&   h
        );
    
    /*------------------------------------------------------------
     * INTERNAL GENERATION FUNCTIONS
     *------------------------------------------------------------*/

    /**
     * @brief generate a Y matrix given the weights and states
     */
    virtual void sample_data_from_prior_(Data_matrix& Y);
    
    //virtual void sample_data_from_prior_(Data_matrix_list& Y);
    
    /*------------------------------------------------------------
     * HYPERPARAMETERS
     *------------------------------------------------------------*/
    const double a_h_;
    const double b_h_;

    /*------------------------------------------------------------
     * STATE VARIABLES
     *------------------------------------------------------------*/
    Precision_vector h_; //!< Noise precision parameter
    
    /*------------------------------------------------------------
     * DATA
     *------------------------------------------------------------*/
    /*
    Data_matrix Y_;
    Data_matrix Y_test_;
     */
    Data_matrix_list Y_;
    Data_matrix_list Y_test_;
};

#endif

