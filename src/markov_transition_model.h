/* $Id: markov_transition_model.h 21265 2017-02-21 15:54:09Z chuang $ */

#ifndef MARKOV_TRANSITION_MODEL_H_
#define MARKOV_TRANSITION_MODEL_H_

/*!
 * @file markov_transition_model.h
 *
 * @author Colin Dawson 
 */

#include "parameters.h"
#include "util.h"
#include "transition_prior.h"
#include "dynamics_model.h"
#include <boost/shared_ptr.hpp>

class Markov_transition_model;
struct Markov_transition_parameters;

typedef boost::shared_ptr<Markov_transition_parameters> Markov_param_ptr;
typedef boost::shared_ptr<Markov_transition_model> Markov_model_ptr;

struct Markov_transition_parameters : public Dynamics_parameters
{
    
    Markov_transition_parameters(
        const Parameters&  params,
        const std::string& name = "Markov_transition_model"
        ) : Dynamics_parameters(params, name)
    {}

    virtual ~Markov_transition_parameters() {}

    virtual Dynamics_model_ptr make_module() const;
};

class Markov_transition_model : public Dynamics_model
{
public:
    typedef Markov_transition_model Self;
    typedef Dynamics_model Base_class;
    typedef kjb::Vector Message_vector;
    typedef std::vector<size_t> Index_list;
    typedef std::vector<Message_vector> Message_matrix;
    typedef std::vector<Index_list> Index_list_list;
    typedef std::vector<Message_matrix> Message_list;
    typedef std::vector<Prob_vector> Prob_list;
    typedef std::vector<Index_list_list> Index_list_list_list;
public:
    /*------------------------------------------------------------
     * CONSTRUCTORS
     *------------------------------------------------------------*/
    
    Markov_transition_model(
        const Markov_transition_parameters* const hyperparams
        ) : Base_class(hyperparams),
            messages_()
    {};

    virtual ~Markov_transition_model() {}

protected:
    virtual void initialize_resources();
    virtual void initialize_params();
    //virtual void sample_labels_from_prior();
    virtual void sample_labels_from_prior(const T_list& T);
    virtual void update_params(const size_t& i, const Likelihood_matrix& llm);
    virtual double get_log_marginal_likelihood(const size_t& i, const Likelihood_matrix& llm);
    virtual double get_test_log_marginal_likelihood(const size_t& i, const Likelihood_matrix& llm);
    void update_z_(const size_t& i, const Likelihood_matrix& llm);
    void update_z_beamily_(const size_t& i);
    
    void pass_messages_forward_(Prob& log_marginal_likelihood_i, const size_t& i, const Likelihood_matrix& llm);
    void pass_test_messages_forward_(Prob& test_log_marginal_likelihood,
                                     const size_t& i,
                                     const Likelihood_matrix& llm);
    void pass_messages_forward_beamily_(Prob& log_marginal_likelihood_i, const size_t& i);
    void sample_z_backwards_(const size_t& i);
    void sample_z_backwards_beamily_(const size_t& i);
    
    /*
    virtual void update_params(const Likelihood_matrix& llm);
    virtual double get_log_marginal_likelihood(const Likelihood_matrix& llm);
    void update_z_(const Likelihood_matrix& llm);
    void update_z_beamily_();
    
    void pass_messages_forward_(Prob& result, const Likelihood_matrix& llm);
    void pass_messages_forward_beamily_(Prob& result);
    void sample_z_backwards_();
    void sample_z_backwards_beamily_();
     */

    /*------------------------------------------------------------
     * AUXILIARY VARIABLES
     *------------------------------------------------------------*/
    Prob_list slice_variables_;
    //Prob_vector slice_variables_;
    
    const Message_vector& messages(const size_t& i, const size_t& t) const {return messages_.at(i).at(t);}
    Message_vector& messages(const size_t& i, const size_t& t) {return messages_.at(i).at(t);}
    //const Message_vector& messages(const size_t& t) const {return messages_.at(t);}
    //Message_vector& messages(const size_t& t) {return messages_.at(t);}
    
    Message_list              messages_;
    Index_list_list_list      slice_indices_;
    //mutable Message_matrix    messages_;
    //mutable Index_list_list   slice_indices_;

};

#endif

