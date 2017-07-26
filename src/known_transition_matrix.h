/* $Id: known_transition_matrix.h 21103 2017-01-22 16:27:51Z chuang $ */

#ifndef KNOWN_TRANSITION_MATRIX_H_
#define KNOWN_TRANSITION_MATRIX_H_

/*!
 * @file known_transition_matrix.h
 *
 * @author Colin Dawson 
 */

#include "transition_prior.h"
#include "parameters.h"
#include <string>

struct Known_transition_parameters :
    public Transition_prior_parameters
{
    const bool uniform_transitions;
    const bool uniform_initial_distribution;
    const std::string transition_matrix_path;
    const std::string initial_distribution_path;
    
    Known_transition_parameters(
        const Parameters& params,
        const std::string& name = "Known_transition_matrix"
        ) : Transition_prior_parameters(params, name),
            uniform_transitions(!params.exists(name, "transition_matrix_path")),
            uniform_initial_distribution(!params.exists(name, "initial_distribution_path")),
            transition_matrix_path(
                uniform_transitions ? std::string() :
                params.get_param_as<std::string>(name, "transition_matrix_path") + "pi"),
            initial_distribution_path(
                uniform_initial_distribution ? std::string() : 
                params.get_param_as<std::string>(name, "initial_distribution_path") + "pi0")
    {}

    virtual ~Known_transition_parameters() {}

    virtual Transition_prior_ptr make_module() const;
};

class Known_transition_matrix : public Transition_prior
{
public:
    typedef Known_transition_matrix Self;
    typedef Known_transition_parameters Params;
    typedef Transition_prior Base_class;
public:
    Known_transition_matrix(
        const Params* const params
        ) : Transition_prior(params),
            uniform_transitions(params->uniform_transitions),
            uniform_initial_distribution(params->uniform_initial_distribution),
            transition_matrix_path(params->transition_matrix_path),
            initial_distribution_path(params->initial_distribution_path)
    {}

    virtual ~Known_transition_matrix() {}

    virtual void initialize_params();
    virtual void input_previous_results(const std::string& input_path, const std::string& name);
    virtual void set_up_results_log() const;
    virtual void write_state_to_file(const std::string& name) const;
protected:
    const bool uniform_transitions;
    const bool uniform_initial_distribution;
    const std::string& transition_matrix_path;
    const std::string& initial_distribution_path;
};

#endif

