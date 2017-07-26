/* $Id: parameters.h 18824 2015-04-15 15:12:54Z clayton $ */

/*!
 * @file parameters.h
 *
 * @author Clayton T. Morrison
 */

#ifndef PARAMETERS_H_
#define PARAMETERS_H_

#include <l_cpp/l_exception.h>

#include <iostream>
#include <sstream>      // std::ostringstream
#include <fstream>      // std::ofstream
#include <string>
#include <tuple>
#include <set>
#include <typeinfo>

#include <boost/unordered_map.hpp>
#include <boost/lexical_cast.hpp>

#define NO_RANDOM_SEED "none"
#define PARAMETERS_DIR "parameters/"
#define PARAMETERS_SUBDIR ""
#define DATA_DIR "data/"
#define RESULTS_DIR "results/"
#define DIAGNOSTICS_DIR "diagnostics/"
#define EVAL_DIR "eval/"

typedef std::vector<int> Define;

bool valid_prob_param(const double& val);
bool valid_shape_param(const double& val);
bool valid_rate_param(const double& val);
bool valid_scale_param(const double& val);
bool valid_beta_param(const double& val);
bool valid_conc_param(const double& val);
bool valid_decay_param(const double& val);
bool valid_mass(const double& val);

template<typename T>
bool any_value(const T&) {return true;}

template<typename T>
std::string get_typename_as_string(){return std::string();}

struct Parameter_value
{
    std::string value_string;
	
    Parameter_value( const std::string & value_string ) : value_string(value_string) { }

    template<typename T>
    T get_param_as() const
    {
        T value;
        value = boost::lexical_cast<T>(value_string);
        return value;
    }
};

struct Parameter_key
{
    const std::string object_name;
    const std::string param_name;

    Parameter_key(const std::string object_name, const std::string param_name)
        : object_name(object_name), param_name(param_name)
    { }
};

inline std::size_t hash_value(const Parameter_key key)
{
    std::size_t seed = 0;
    boost::hash_combine(seed, key.object_name);
    boost::hash_combine(seed, key.param_name);
    return seed;
}

inline bool operator==(const Parameter_key & lhs, const Parameter_key rhs)
{
    // compare returns 0 when the two strings are equal!
    return ( (lhs.object_name.compare(rhs.object_name) == 0) &&
             (lhs.param_name.compare(rhs.param_name) == 0) );
}

// storage of parameter instances
typedef boost::unordered_map< Parameter_key , Parameter_value > Parameter_map;

// storage of defined parameters, used for iteration
typedef std::set<std::string> Defined_names;
typedef Defined_names Defined_objects;
typedef boost::unordered_map< std::string , Defined_names > Defined_params_for_object;

// Parameters class

class Parameters
{
public:

    /*------------------------------------------------------------
     * STORAGE
     *------------------------------------------------------------*/

    // storage for parameter value instances
    Parameter_map parameter_map;

    // bookkeeping defined parameters, used for iteration
    Defined_objects defined_objects;
    Defined_params_for_object defined_params_for_object;

    /*------------------------------------------------------------
     * CONSTRUCTORS
     *------------------------------------------------------------*/

    Parameters()
    {
    	add_string(":environment","parameters_dir",PARAMETERS_DIR);
    	add_string(":experiment","parameters_subdir",PARAMETERS_SUBDIR);
    	add_string(":environment","data_dir",DATA_DIR);
    	add_string(":environment","results_dir",RESULTS_DIR);
    	add_string(":environment","diagnostics_dir",DIAGNOSTICS_DIR);
    	add_string(":environment","eval_dir",EVAL_DIR);

        add_string(":experiment","random_seed",NO_RANDOM_SEED);
    }

    virtual ~Parameters() {}

    /*------------------------------------------------------------
     * TOP-LEVEL INTERFACE
     *------------------------------------------------------------*/

    void read(const std::string & infile, const bool verbose = false);

    void save(const std::string & outfile);

    void output(std::ostream & out) const;

    template<typename T>
    void add(const std::string & object_name, const std::string & param_name, const T & value)
    {
        std::ostringstream convert;
        convert << value;
        std::string value_string = convert.str();
        add_string( object_name, param_name, value_string);
    }

    void add_string(const std::string & object_name, const std::string & param_name, const std::string & value_string);

    bool exists(const std::string & object_name, const std::string & param_name) const;

    template<typename T>
    T get_param_as(
        const std::string& object,
        const std::string& param,
        const std::string& failure_message = "",
        bool (*condition)(const T&) = any_value<T>
        ) const;
};

template<typename T>
T Parameters::get_param_as(
    const std::string& object,
    const std::string& param,
    const std::string& failure_message,
    bool (*condition)(const T&)
    ) const
{
    Parameter_map::const_iterator it =
        parameter_map.find(Parameter_key(object, param));
    std::string msg =
        failure_message.empty()
        ? "I/O ERROR: Could not find parameter @ '" + object + "' '" + param + "'"
        : failure_message;
    IFT(it != parameter_map.end(), kjb::IO_error, msg);
    Parameter_value entry = (*it).second;
    try
    {
        T result = entry.get_param_as<T>();
        IFT(condition(result), kjb::Illegal_argument, failure_message);
        return result;
    } catch(boost::bad_lexical_cast&) {
        KJB_THROW_3(
            kjb::IO_error,
            "I/O ERROR: Parameter @'%s' '%s' has value %s, but is required "
            " to be interpretable as type %s",
            (object.c_str())(param.c_str())
            (entry.value_string.c_str())(get_typename_as_string<T>().c_str())
            );
    }
}

#endif // PARAMETERS_H_

