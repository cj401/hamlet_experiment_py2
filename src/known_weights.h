/* $Id: known_weights.h 21103 2017-01-22 16:27:51Z chuang $ */

#ifndef KNOWN_WEIGHTS_H_
#define KNOWN_WEIGHTS_H_

/*!
 * @file emission_known_weights.h
 *
 * @author Colin Dawson 
 */

#include "parameters.h"
#include "weight_prior.h"
#include <m_cpp/m_matrix.h>
#include <string>

class Known_weights;

 struct Known_weights_parameters : public Weight_prior_parameters
{
    typedef Weight_prior_parameters Base_class;
    const std::string weights_file;

    Known_weights_parameters(
        const Parameters& params,
        const std::string& name = "Known_weights"
        );

    virtual ~Known_weights_parameters() {}

    virtual Weight_prior_ptr make_module() const;
};

class Known_weights : public Weight_prior
{
public:
    typedef Known_weights Self;
    typedef Known_weights_parameters Params;
    typedef Weight_prior Base_class;
public:
    Known_weights(
        const Params* const hyperparameters
        ) : Base_class(hyperparameters),
            weights_file(hyperparameters->weights_file)
    {}

    virtual ~Known_weights() {}
    
    virtual void initialize_params();
    
    virtual void input_previous_results(const std::string&, const std::string&) {};
    
    virtual void generate_data(const std::string& filename);
    
    virtual void update_params() {};

    virtual Weight_vector propose_weight_vector(const size_t&) const
    {
        return Weight_vector((int) K(),0.0);
    }

    virtual void write_state_to_file(const std::string& name) const;

    const std::string weights_file;
};

#endif
