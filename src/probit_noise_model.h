
/* $Id: probit_noise_model.h 21245 2017-02-13 18:27:33Z chuang $ */

#ifndef PROBIT_NOISE_MODEL_H_
#define PROBIT_NOISE_MODEL_H_

/*!
 * @file probit_noise_model.h
 *
 * @author Colin Dawson 
 */

#include "parameters.h"
#include "util.h"
#include "noise_model.h"
#include <m_cpp/m_vector.h>
#include <m_cpp/m_matrix.h>
#include <m_cpp/m_mat_view.h>
#include <string>

/* /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ /\ */

class Probit_noise_model;

struct Probit_noise_parameters : public Noise_model_parameters
{
    /*
    Probit_noise_parameters(
        const Parameters&,
        const std::string& = "Probit_noise_model"
        )
    {}
     */
    
    Probit_noise_parameters(
        const Parameters& params,
        const std::string& name = "Probit_noise_model"
        ) : Noise_model_parameters(params, name)
    {}

    virtual ~Probit_noise_parameters() {}

    virtual Noise_model_ptr make_module() const;
};

class Probit_noise_model : public Noise_model
{
public:
    typedef Probit_noise_model Self;
    typedef Noise_model Base_class;
    typedef kjb::Int_matrix Data_matrix;
    typedef kjb::Int_vector Data_vector;
    typedef std::vector<Data_matrix> Data_matrix_list;
    typedef Noise_model::Noisy_data_matrix Latent_data_matrix;
    typedef Noise_model::Noisy_data_matrix_list Latent_data_matrix_list;
public:

    /*------------------------------------------------------------
     * CONSTRUCTORS
     *------------------------------------------------------------*/

    /**
     * @brief construct from a collection of hyperparameters
     */
    /*
    Probit_noise_model() :
        Ystar_(), Y_()
    {}
     */
    Probit_noise_model(
        const Probit_noise_parameters* const hyperparameters
        ) : Noise_model(hyperparameters),
            Ystar_(), Y_()
            {}
    
    /*------------------------------------------------------------
     * DESTRUCTOR
     *------------------------------------------------------------*/
    /**
     * @brief virtual destructor
     */
    virtual ~Probit_noise_model() {}

    /*------------------------------------------------------------
     * INITIALIZERS
     *------------------------------------------------------------*/
    /**
     * @brief allocate memory for parameter objects
     */
    void initialize_resources();

    /**
     * @brief initialize values of the parameter objects
     */
    void initialize_params();
    
    /**
     * @brief input information from previous results
     */
    void input_previous_results(const std::string& input_path, const std::string& name);
    
    /**
     * @brief read in a dataset from a file
     */
    void add_data(const std::string& filename, const size_t& num_files);

    /**
     * @brief read in a test set from a file
     */
    void add_test_data(const std::string& filename, const size_t& num_files);
    
    /**
     * @brief generate a dataset and write it to a file
     */
    void generate_data(const std::string& filename);

    /**
     * @brief generate a dataset and write it to a file
     */
    void generate_test_data(const std::string& filename);
    
    /*------------------------------------------------------------
     * INFERENCE INTERFACE
     *------------------------------------------------------------*/
    /**
     * @brief do a Gibbs update step
     */
    void update_params();

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
     * ACCESSORS NEEDED BY OTHER MODULES
     *------------------------------------------------------------*/

    Latent_data_matrix_list noisy_data() const {return Ystar_;}
    
    Latent_data_matrix noisy_data(const size_t& i) const {return Ystar_.at(i);}
    /*
    Latent_data_matrix noisy_data() const {return Ystar_;}
     */

    Noise_parameters parameters() const
    {
        static Noise_parameters p(1, (int) K(), 1.0);
        return p;
    }

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
     * INTERNAL ACCESSORS
     *------------------------------------------------------------*/

    /**
     * @brief access the latent data matrix
     */
    /*
    const Latent_data_matrix& Ystar() const {return Ystar_;}
    int Y(const size_t& t, const size_t& k) const {return Y_.at(t,k);}
     */
    const Latent_data_matrix& Ystar(const size_t i) const {return Ystar_[i];}
    int Y(const size_t& i, const size_t& t, const size_t& k) const {return Y_[i].at(t,k);}

    /*------------------------------------------------------------
     * LVALUE ACCESSORS
     *------------------------------------------------------------*/
    /**
     * @brief access an lvalue element of the latent data matrix
     */
    
    double& Ystar(const size_t& i, const size_t& t, const size_t& k) {return Ystar_.at(i).at(t,k);}
    int& Y(const size_t& i, const size_t& t, const size_t& k) {return Y_.at(i).at(t,k);}
    
    /*
    double& Ystar(const size_t& t, const size_t& k) {return Ystar_.at(t,k);}
    int& Y(const size_t& t, const size_t& k) {return Y_.at(t,k);}
     */

    /*------------------------------------------------------------
     * INTERNAL INFERENCE FUNCTIONS
     *------------------------------------------------------------*/
    /**
     * @brief resample (using Gibbs) the latent data matrix
     */
    virtual void update_Ystar_();
    void update_Ystar_(const size_t& i);

    /**
     * @brief get the likelihood of observation at time t given mean 'mean'
     */
    double log_likelihood(const size_t& i, const size_t& t, const Mean_vector& mean) const;
    /*
    double log_likelihood(const size_t& t, const Mean_vector& mean) const;
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

    /*------------------------------------------------------------
     * INTERNAL GENERATION FUNCTIONS
     *------------------------------------------------------------*/
    /**
     * @brief generate a Y matrix given the weights and states
     */
    virtual void sample_data_from_prior_(
        Data_matrix&        observed,
        Latent_data_matrix& latent);
    
    //virtual void sample_data_from_prior_(
       // Data_matrix_list&        observed,
       // Latent_data_matrix_list& latent);
    
    /*
    Latent_data_matrix Ystar_;
     */
    Latent_data_matrix_list Ystar_;
    Data_matrix_list Y_;
    Data_matrix_list Y_test_;
    /*
    Data_matrix Y_;
    Data_matrix Y_test_;
     */
};

#endif
