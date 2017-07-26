/* $Id: parameters.cpp 18824 2015-04-15 15:49:03Z clayton $ */

/*!
 * @file parameters.cpp
 *
 * @author Clayton T. Morrison
 */

#include "parameters.h"
#include <l_cpp/l_exception.h>
#include <boost/lexical_cast.hpp>

// Parameters

void Parameters::read(const std::string & infile, const bool verbose)
{

    // either comment '//...'
    // or 3 space-separated tokens optionally followed by comment

    std::string line;
    std::ifstream file (infile);
    if ( file.is_open() )
    {
        size_t line_number = 1;
        while ( getline( file , line ) )
        {
            size_t idx = 0;
            size_t start = 0;
            bool in_token = false;
            std::string object_string;
            bool object_string_done = false;
            std::string param_string;
            bool param_string_done = false;
            std::string value_string;
            bool value_string_done = false;
            for ( std::string::iterator it = line.begin(); 
                  ( it != line.end() ) && ( *it != '/' ); 
                  ++it )
            {
                if (verbose) { std::cout << "[" << *it << "]"; }
                if ( *it != ' ' )
                {
                    if ( !in_token )
                    {
                        // non-blank and not in token, so start token
                        start = idx;
                        in_token = true;
                    }
                }
                else if ( *it == ' ' && in_token )
                {
                    // was in token and reached a space, so end of token
                    if ( !object_string_done )
                    {
                        object_string_done = true;
                        object_string = line.substr(start, (idx - start));
                        in_token = false;
                    }
                    else if ( !param_string_done )
                    {
                        param_string_done = true;
                        param_string = line.substr(start, (idx - start));
                        in_token = false;
                    }
                    else if ( !value_string_done )
                    {
                        value_string_done = true;
                        value_string = line.substr(start, (idx - start));
                        in_token = false;
                    }
                    else
                    {
                        std::cout << "WARNING: Line " << line_number << " has additional characters..." << std::endl;
                        std::cout << ">>> '" << line << "'" << std::endl;
                    }
                }

                ++idx;
            } // end looping over line

            if (verbose) { std::cout << std::endl; }

            // close the last token as a value_string, if object and param defined
            if ( object_string_done && param_string_done && !value_string_done )
            {
                value_string_done = true;
                value_string = line.substr(start);
                in_token = false; // not needed...
            }

            if ( object_string_done && param_string_done && value_string_done )
            {
                if(verbose)
                {
                    std::cout << "line " << line_number << " : "
                              << "o='" << object_string << "' "
                              << "p='" << param_string << "' "
                              << "v='" << value_string << "' "
                              << std::endl;
                }

                // Add parameter
                add(object_string, param_string, value_string);
            }
            else if ( ( object_string_done || param_string_done ) && !value_string_done )
            {
                std::cout << "WARNING: Line " << line_number << " is ill-formed (partially specified) -- SKIPPING" << std::endl;
                std::cout << ">>> '" << line << "'" << std::endl;
            }

            ++line_number;
        }
        file.close();
    }
    else
    {
        std::cout << "Unable to open parameters file: '" 
                  << infile << "'" << std::endl;
    }
}

void Parameters::save(const std::string & outfile)
{
    // TODO: add error checking!
    std::ofstream file;
    file.open( outfile );
    output( file );
    file.close();
}

void Parameters::output(std::ostream & out) const
{
    out << "// Parameters" << std::endl;
    for ( Defined_objects::const_iterator object_name_it = Parameters::defined_objects.begin();
          object_name_it != Parameters::defined_objects.end();
          ++object_name_it )
    {
        out << std::endl;
        Defined_params_for_object::const_iterator param_name_set_it = Parameters::defined_params_for_object.find( *object_name_it );
        if ( param_name_set_it != Parameters::defined_params_for_object.end() )
        {
            out << "// " << *object_name_it << " parameters" << std::endl;
            Defined_names parameter_name_set = (*param_name_set_it).second;
            for ( Defined_names::const_iterator param_name_it = parameter_name_set.begin();
                  param_name_it != parameter_name_set.end();
                  ++param_name_it )
            {
                Parameter_key pk( *object_name_it , *param_name_it );
                Parameter_map::const_iterator pvalue_it = Parameters::parameter_map.find( pk );
                if ( pvalue_it != Parameters::parameter_map.end() )
                {
                    out << *object_name_it << " " << *param_name_it << " " << (*pvalue_it).second.value_string << std::endl;
                }
            }
        }
    }
}

void Parameters::add_string(
    const std::string& object_name,
    const std::string& param_name,
    const std::string& value_string
    )
{
    Parameter_key pk( object_name , param_name );
    Parameter_value pv( value_string );
    if(exists(object_name, param_name))
    {
        Parameter_map::iterator it = parameter_map.find(pk);
        std::cerr << "Overriding parameter " << object_name << ":" << param_name
                  << " with value " << value_string << std::endl;
        it->second = pv;
        return;
    }

    Defined_objects::iterator def_obj_it = defined_objects.find(object_name);
    if(def_obj_it == defined_objects.end())
    {
        // add new object category
        defined_objects.insert( object_name );
        Defined_names new_param_name_set;
        new_param_name_set.insert( param_name );
        defined_params_for_object.insert( std::make_pair( object_name , new_param_name_set ) );
    }
    Defined_params_for_object::iterator param_name_set = defined_params_for_object.find( object_name );
    Defined_names::iterator param_names_it = ((*param_name_set).second).find(param_name);
    if(param_names_it == ((*param_name_set).second).end())
    {
        ((*param_name_set).second).insert( param_name );
    }
    parameter_map.insert( std::make_pair( pk , pv ) );
}

bool valid_prob_param(const double& val)
{
    return val >= 0.0 && val <= 1.0;
}

bool valid_shape_param(const double& val)
{
    return val > 0;
}

bool valid_rate_param(const double& val)
{
    return val > 0;
}

bool valid_scale_param(const double& val)
{
    return val > 0;
}

bool valid_beta_param(const double& val)
{
    return val > 0;
}

bool valid_conc_param(const double& val)
{
    return val > 0;
}

bool valid_decay_param(const double& val)
{
    return val >= 0;
}

bool valid_mass(const double& val)
{
    return val >= 0;
}

bool Parameters::exists(const std::string & object_name, const std::string & param_name) const
{
    Parameter_key pk( object_name , param_name );
    Parameter_map::const_iterator pvalue_it = Parameters::parameter_map.find( pk );
    return pvalue_it != Parameters::parameter_map.end();
}

template<>
std::string get_typename_as_string<std::string>()
{
    return "std::string";
}

template<>
std::string get_typename_as_string<bool>()
{
    return "bool";
}

template<>
std::string get_typename_as_string<double>()
{
    return "double";
}

template<>
std::string get_typename_as_string<int>()
{
    return "int";
}

template<>
std::string get_typename_as_string<unsigned int>()
{
    return "unsigned int";
}

template<>
std::string get_typename_as_string<size_t>()
{
    return "size_t";
}

