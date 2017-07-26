/* $Id: categorical_noise_model.h 21245 2017-02-13 18:27:33Z chuang $ */

#ifndef CATEGORICAL_NOISE_MODEL_H_
#define CATEGORICAL_NOISE_MODEL_H_

/*!
 * @file categorical_noise_model.h
 *
 * @author Colin Dawson 
 */

#include "noise_model.h"
#include "parameters.h"

class Categorical_noise_model;
struct Categorical_noise_model_parameters;

struct Categorical_noise_model_parameters : public Noise_model_parameters
{
    const size_t K; //!< number of possible categories in output

    Categorical_noise_model_parameters(
        const Parameters& params,
        const std::string& name = "Dirichlet_multinomial_emissions"
        );

    virtual ~Categorical_noise_model_parameters() {}

    static const std::string& bad_K()
    {
        static const std::string msg =
            "I/O ERROR: For a Categorical emission model, config file must specify"
            " a positive integer K, representing the number of possible observation"
            " categories";
        return msg;
    }

    virtual Noise_model_ptr make_module() const;
};

class Categorical_noise_model : public Noise_model
{
public:
    /*------------------------------------------------------------
     * TYPEDEFS
     *------------------------------------------------------------*/
    typedef Categorical_noise_model Self;
    typedef Noise_model Base_class;
    typedef kjb::Int_vector Data_vector;
    typedef kjb::Int_matrix Data_matrix;
    typedef std::vector<Data_matrix> Data_matrix_list;
    using Base_class::T;
    typedef Base_class::Noisy_data_matrix Noisy_data_matrix;
    typedef Base_class::Noisy_data_matrix_list Noisy_data_matrix_list;

public:
    /*------------------------------------------------------------
     * CONSTRUCTORS
     *------------------------------------------------------------*/

    /*
    Categorical_noise_model(
        const Categorical_noise_model_parameters* const hyperparameters
        ) : K_(hyperparameters->K) {}
     */
    Categorical_noise_model(
        const Categorical_noise_model_parameters* const hyperparameters
        ) : Noise_model(hyperparameters),
            K_(hyperparameters->K)
            {}

    /*------------------------------------------------------------
     * DESTRUCTOR
     *------------------------------------------------------------*/

    ~Categorical_noise_model() {}

    /*------------------------------------------------------------
     * INITIALIZERS
     *------------------------------------------------------------*/

    virtual void initialize_resources();

    virtual void initialize_params();

    virtual void input_previous_results(const std::string& input_path, const std::string& name);

    virtual void add_data(const std::string& filename, const size_t& num_files);

    virtual void add_test_data(const std::string& filename, const size_t& num_files);

    virtual void generate_data(const std::string& path);

    virtual void generate_test_data(const std::string& path);

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
     * ACCESSORS NEEDED BY OTHER MODULES
     *------------------------------------------------------------*/
    
    Noisy_data_matrix_list noisy_data() const
    {
        Noisy_data_matrix_list Y_noisy(NF());
        for (size_t i = 0; i < NF(); i++)
        {
            Y_noisy.at(i) = noisy_data(i);
        }
        return Y_noisy;
    }
    
    Noisy_data_matrix noisy_data(const size_t& i) const {return Y_.at(i);}
    
    /*
    Noisy_data_matrix noisy_data() const {return Y_;}
     */

    Noise_parameters parameters() const;

    /*------------------------------------------------------------
     * COMPUTATION NEEDED BY OTHER MODULES
     *------------------------------------------------------------*/
    virtual double log_likelihood_ratio_for_state_change(
        const size_t&           i,
        const Mean_vector&      x,
        const Mean_vector&      delta,
        const kjb::Index_range& times
        );
    
protected:
    /*------------------------------------------------------------
     * INTERNAL COMPUTATION
     *------------------------------------------------------------*/
    
    /*
    virtual double log_likelihood(const size_t& t, const Mean_vector& mean) const;
    virtual double test_log_likelihood(const size_t& t, const Mean_vector& mean) const;
     */
    virtual double log_likelihood(const size_t& i, const size_t& t, const Mean_vector& mean) const;

    virtual double test_log_likelihood(const size_t& i, const size_t& t, const Mean_vector& mean) const;

    static double log_likelihood(
        const Data_matrix&        Y,
        const size_t&             t,
        const Mean_vector&        mean
        );

    void sample_data_(Data_matrix& Y);
    
    //void sample_data_(Data_matrix_list& Y);

    /*------------------------------------------------------------
     * DIMENSION VARIABLES
     *------------------------------------------------------------*/
    const size_t K_;

    /*------------------------------------------------------------
     * DATA
     *------------------------------------------------------------*/

    Data_matrix_list Y_;
    Data_matrix_list Y_test_;
    /*
     Data_matrix Y_;
     Data_matrix Y_test_;
     */
    
};

#endif

