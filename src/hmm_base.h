/* $Id: hmm_base.h 21443 2017-06-28 18:13:50Z chuang $ */

#ifndef HMM_BASE_H_
#define HMM_BASE_H_

/*!
 * @file hmm_base.h
 *
 * @author Colin Dawson 
 */

#include "util.h"
#include "emission_model.h"

class Emission_model;

class HMM_base
{
public:
    friend class Emission_model;
public:
    /*------------------------------------------------------------
     * CONSTRUCTORS
     *------------------------------------------------------------*/
    HMM_base(
        const std::string& write_path,
        const Emission_param_ptr emission_parameters
        ) : test_data_exists(false),
            write_path(write_path),
            emission_model_(emission_parameters->make_module())
    {}

    HMM_base(
        const std::string& write_path,
        const Emission_model_ptr emission_model
        ) : test_data_exists(false),
            write_path(write_path),
            emission_model_(emission_model)
    {}

    /*------------------------------------------------------------
     * DESTRUCTOR
     *------------------------------------------------------------*/
    ~HMM_base();

    void add_data(const std::string& path, const size_t& num_files = 1);

    void add_test_data(const std::string& path, const size_t& num_files = 1);

    void generate_data(
        const size_t&      num_sequences,
        const size_t&      T,
        const size_t&      K,
        const std::string& name
        );
    
    virtual void generate_test_sequence(
        const size_t&      num_sequences,
        const size_t&      T,
        const std::string& name
        );

    virtual void resample() = 0;

    virtual void set_up_results_log() const = 0;

    virtual void input_previous_results(const std::string& input_path, const std::string& name);

    virtual void write_state_to_file(const std::string& name) const;

    virtual void add_ground_truth_eval_header() const {}

    virtual void compare_state_sequence_to_ground_truth(
        const std::string&,
        const std::string&
        ) const {}
    /*------------------------------------------------------------
     * DIMENSION ACCESSORS
     *------------------------------------------------------------*/
    /*
    const size_t& T() const {return T_;}
     */
    const size_t& NF() const {return NF_;}
    const size_t& test_NF() const {return test_NF_;}
    const T_list& T() const {return T_;}
    size_t T(const size_t& i) const {return T_.at(i);}
    
    const size_t& K() const {return K_;}
    const size_t& D() const {return D_;}
    virtual const size_t& J() const
    {
        IFT(false, kjb::Not_implemented,
            "Attempted to fetch J parameter for a model where this is"
            "undefined");
    }
    
    virtual size_t D_prime() const = 0;
    //const size_t& test_T() const {return test_T_;}
    const T_list& test_T() const {return test_T_;}
    size_t test_T(const size_t& i) const {return test_T_.at(i);}

    /*------------------------------------------------------------
     * ACCESSORS TO SHARED VARIABLES
     *------------------------------------------------------------*/
    const State_matrix& theta_star(const size_t& i) const
    {
        return theta_star_[i];
    }
    
    State_matrix& theta_star(const size_t& i)
    {
        return theta_star_[i];
    }
    
    const State_matrix_list& theta_star() const
    {
        return theta_star_;
    }
    /*
    const State_matrix& theta_star() const {return theta_star_;}
    */

    State_matrix_list& theta_star() {return theta_star_;}
    
    virtual const Time_set& partition_map(const size_t&, const State_indicator&) const
    {
        IFT(false, kjb::Not_implemented,
            "Attempting to retrieve partition map for a model type"
            "that does not support it");
    }

    virtual size_t dimension_map(const size_t& dprime) const
    {
        return dprime;
    }
    
    /*------------------------------------------------------------
     * VERBOSE SET UP FUNCTIONS
     *------------------------------------------------------------*/
    virtual void set_up_verbose_level(const size_t verbose);

    /*------------------------------------------------------------
     * INTERNAL INFERENCE FUNCTIONS
     *------------------------------------------------------------*/
public:
    virtual void initialize_parent_links() = 0;
protected:
    virtual void initialize_resources_();
    virtual void initialize_params_();
    // void initialize_partition_();
    virtual void sync_theta_star_() = 0;
public:
    /*------------------------------------------------------------
     * EXPERIMENT STATE VARIABLES
     *------------------------------------------------------------*/
    bool test_data_exists;
    std::string write_path;
protected:
    /*------------------------------------------------------------
     * DIMENSION VARIABLES
     *------------------------------------------------------------*/
    size_t NF_;
    size_t test_NF_;
    
    size_t D_;
    T_list T_;
    T_list test_T_;
    /*
    size_t T_;
    size_t test_T_;
     */
    size_t K_;
    
    /*------------------------------------------------------------
     * LINKS TO COMPONENT MODULES
     *------------------------------------------------------------*/
    Emission_model_ptr emission_model_;

    /*------------------------------------------------------------
     * SHARED VARIABLES
     *------------------------------------------------------------*/
    State_matrix_list theta_star_;
    /*
    State_matrix theta_star_;
     */
    
    /*------------------------------------------------------------
     * VERBOSE VARIABLES
     *------------------------------------------------------------*/
    size_t verbose_;
};

#endif
