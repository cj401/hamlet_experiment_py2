
#include "program_options_and_parameters.h"

// grabs ERE
// l/l_sys_io functions are in kjb_c namespace
// included here for:
//    int is_directory(const char *path)
//    int kjb_mkdir(const char *dir_path)
#include <l_cpp/l_stdio_wrap.h>
#include <l_cpp/l_util.h>
#include <prob_cpp/prob_sample.h>     // kjb::DEFAULT_SEED

#include <boost/program_options.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/format.hpp>

#include <string>
#include <iostream>
#include <sys/time.h>
#include <ctime>

#include "util.h"

std::string get_timestamp_string(size_t verbose)
{
    // time_t now = time(0);
    // tm *ltm = localtime(&now);

    struct timeval tv;
    struct tm *ltm;
    gettimeofday(&tv, NULL);
    ltm = localtime(&tv.tv_sec);

    boost::format year_fmt = boost::format("%04d") % boost::lexical_cast<int>(1900 + ltm->tm_year);
    std::string year_str(year_fmt.str());
    boost::format month_fmt = boost::format("%02d") % boost::lexical_cast<size_t>(1 + ltm->tm_mon);
    std::string month_str(month_fmt.str());
    boost::format day_fmt = boost::format("%02d") % boost::lexical_cast<size_t>(ltm->tm_mday);
    std::string day_str(day_fmt.str());
    boost::format hour_fmt = boost::format("%02d") % boost::lexical_cast<size_t>(1 + ltm->tm_hour);
    std::string hour_str(hour_fmt.str());
    boost::format minute_fmt = boost::format("%02d") % boost::lexical_cast<size_t>(1 + ltm->tm_min);
    std::string minute_str(minute_fmt.str());
    boost::format second_fmt = boost::format("%02d") % boost::lexical_cast<size_t>(1 + ltm->tm_sec);
    std::string second_str(second_fmt.str());
    boost::format usec_fmt = boost::format("%03d") % boost::lexical_cast<size_t>(tv.tv_usec);
    std::string microseconds_str(usec_fmt.str());

    std::string timestamp_string = year_str + month_str + day_str + "_" + hour_str + minute_str + second_str + microseconds_str;

    if (verbose > 0)
    {
        std::cout << "Year:  '" << year_str << "'" << std::endl;
        std::cout << "Month: '" << month_str << "'" << std::endl;
        std::cout << "Day:   '" << day_str << "'" << std::endl;
        std::cout << "Time:  '" << hour_str << ":";
        std::cout << minute_str << ":";
        std::cout << second_str << "'" << std::endl;
        std::cout << "timestamp_string: '" << timestamp_string << "'" << std::endl;
    }

    return timestamp_string;
}

unsigned int get_usec_random_seed()
{
    struct timeval tv;
    struct tm *ltm;
    gettimeofday(&tv, NULL);
    return ((kjb::DEFAULT_SEED % 1000) * 1000000) + tv.tv_usec;
}

int ensure_directory_exists(const std::string directory_path,
                            const size_t verbose)
{

    // If directory_path does not exist, create it!
    if(kjb_c::is_directory(directory_path.c_str()) != true)
    {
        PMWP(verbose > 1,
             "Cannot find existing directory path: '%s'\nCreating!\n",
             (directory_path.c_str()));
        /*
        if(verbose)
        {
            std::cout << "Cannot find existing directory path: '"
                      << directory_path << "'" << std::endl
                      << "Creating!" << std::endl;
        }
         */
        
        {
            // The KJB macro wraps the argument with 'using namespace kjb_c;' -- in /l_cpp/l_util.h
            KJB(ETX(kjb_c::kjb_mkdir(directory_path.c_str())));
        }

        if (kjb_c::is_directory(directory_path.c_str()) != true)
        {
            std::cerr << "Cannot find directory path after call to"
                      << "kjb_c::kjb_mkdir(\""
                      << directory_path << "\")" << std::endl;
            std::cerr << "Exiting.";
            return kjb_c::ERROR;
        }
        else
        {
            PMWP(verbose > 1,
                 "Successfully created directory path: '%s'\n",
                 (directory_path.c_str()));
            /*
            if(verbose)
            {
                std::cout << "Successfully created directory path: '"
                          << directory_path << "'" << std::endl;
            }
             */
        }
    }
    else
    {
        PMWP(verbose > 1,
             "Found existing directory path: '%s'\n",
             (directory_path.c_str()));
        /*
        if(verbose)
        {
            std::cout << "Found existing directory path: "
                      << directory_path << std::endl;
        }
         */
    }

    return kjb_c::NO_ERROR;
}

int process_program_options_and_read_parameters(
    int argc,
    char *argv[],
    Parameters& params,
    bool& generate,
    size_t& verbose,
    bool& resume
    )
{
    size_t generate_sequence_length = 0;
    size_t generate_observables_dimension = 0;
    std::string random_seed = params.get_param_as<std::string>(":experiment","random_seed");
    size_t random_seed_value;
    std::string parameters_dir = params.get_param_as<std::string>(":environment","parameters_dir");
    std::string parameters_subdir =
        params.get_param_as<std::string>(":experiment","parameters_subdir") + "/";
    
    const std::string parameters_dir_docstring =
        "Parameters directory path\n(default: '" + parameters_dir + "')\n";
    
    std::string parameters_file_name;
    std::string data_dir = params.get_param_as<std::string>(":environment","data_dir");
    
    const std::string data_dir_docstring =
        "[generation or inference] Directory root path of data subdirectory(ies) (gen=write, inf=read)\n(default: '"
        + data_dir + "')\n";

    std::string data_timestamp; // default to empty string, indicating to generate new timestamp
    std::string results_timestamp;
    std::string results_subdir;
    std::string data_file_name;
    // only specified for inference, although used in generation as composition of
    // data_file_name and timestamp
    std::string data_subdir; 
    std::string data_path;
    std::string results_dir = params.get_param_as<std::string>(":environment","results_dir");

    const std::string results_dir_docstring =
        "Output results directory path\n(default: '" + results_dir + "')";

    std::string results_file_name;
    std::string results_path;

    std::string eval_dir = params.get_param_as<std::string>(":environment","eval_dir");

    const std::string eval_dir_docstring =
        "Evaluation directory path\n(default: '" + eval_dir + "')\n";

    std::string weights_file;

    size_t iterations;

    size_t resume_iteration;
    std::string resume_subdir;
    std::string resume_path;
    bool resume_with_old_parameter = false;

    // NOTE: generate sequence has to be triggered by program options (-g)
    // Don't want to specify whether to generate in a parameter file as then
    // saved parameter settings will include, and could accidentally clobber
    // already-generated data.
    
    namespace po = boost::program_options;
    po::options_description odesc("Options");
    odesc.add_options()
        ("help,h", "Print help message")
        ("verbose,v", po::value<size_t>(&verbose),
            "Run verbosely on different levels: 0, 1, 2\n")

        ("parameters,p", po::value<std::string>(&parameters_file_name),
            "Parameters file name")
        ("parameters_dir", po::value<std::string>(&parameters_dir),
            parameters_dir_docstring.c_str())
        ("parameters_subdir", po::value<std::string>(&parameters_subdir),
            parameters_dir_docstring.c_str())

        ("data,d", po::value<std::string>(&data_file_name), 
            "[generation only] Base file name for data")
        ("data_dir", po::value<std::string>(&data_dir),
            data_dir_docstring.c_str())
    
        ("resume_itr", po::value<size_t>(&resume_iteration),
            "[inference only] Resume from which iteration number")
        ("resume_subdir", po::value<std::string>(&resume_subdir),
            "[inference only] inputting results file subdir for resume")
    
        ("results,r", po::value<std::string>(&results_file_name),
            "[inference only] Output results base file name")
        ("results_dir", po::value<std::string>(&results_dir),
         results_dir_docstring.c_str())
        ("results_timestamp", po::value<std::string>(&results_timestamp),
         "[inference only] Manually specify the timestamp for results; when not specified, "
            "timestamp is generated")
        ("data_subdir", po::value<std::string>(&data_subdir),
            "[inference only] Data subdirectory name (to use for reading data for inference)")
        ("iterations,i", po::value<size_t>(&iterations), 
            "[inference only] Number of Gibbs sample iterations")

        ("weights_file", po::value<std::string>(&weights_file),
            "Manually specify .weights file (usually path to .weights file from source data)\n")

        ("generate,g", "Generate new sequence")
        ("sequence,s", po::value<size_t>(&generate_sequence_length),
            "Length for sequence of new generated data (ignored unless -g is specified)")
        ("observable_dimension,k", po::value<size_t>(&generate_observables_dimension),
            "Dimension of each observation vector (ignored unless -g is specified)")
        ("timestamp", po::value<std::string>(&data_timestamp),
            "[generation only] Manually specify the timestamp for generated data directory name;"
            " when not specified, timestamp is generated\n")

        ("seed", po::value<std::string>(&random_seed),
            "Set the random seed;\n(default: none, uses get_usec_random_seed(): "
                "((kjb::DEFAULT_SEED % 1000) * 1000000) + gettimeofday tv_usec ; "
                "cycles every 1000 seconds)");

    po::variables_map vm;
    try
    {
        po::store(po::parse_command_line(argc, argv, odesc), vm); // can throw
        if( vm.count("help") )
        {
            std::cout << "(hdp-)h(s)mm(-lt) model" << std::endl
                      << odesc << std::endl;

            /*
            std::cout << "kjb::DEFAULT_SEED      " << kjb::DEFAULT_SEED << std::endl;
            std::cout << "kjb::DEFAULT_SEED      " << kjb::DEFAULT_SEED << std::endl;
            std::cout << "kjb::DEFAULT_SEED      " << kjb::DEFAULT_SEED << std::endl;
            std::cout << "kjb::DEFAULT_SEED      " << kjb::DEFAULT_SEED << std::endl;
            std::cout << "kjb::DEFAULT_SEED      " << kjb::DEFAULT_SEED << std::endl;
            std::cout << "kjb::DEFAULT_SEED      " << kjb::DEFAULT_SEED << std::endl;

            std::cout << "get_usec_random_seed() " << get_usec_random_seed() << std::endl;
            std::cout << "get_usec_random_seed() " << get_usec_random_seed() << std::endl;
            std::cout << "get_usec_random_seed() " << get_usec_random_seed() << std::endl;
            std::cout << "get_usec_random_seed() " << get_usec_random_seed() << std::endl;
            std::cout << "get_usec_random_seed() " << get_usec_random_seed() << std::endl;
            std::cout << "get_usec_random_seed() " << get_usec_random_seed() << std::endl;
            */

            return STOP_ON_HELP;
        }

        po::notify(vm); // can throw, so do after help in case problem

        if( vm.count("verbose") )
        {
            params.add(":experiment", "verbose", verbose);
            std::cout << "Running on verbose level " << verbose << std::endl;
            /*
            std::cout << "Running verbose" << std::endl;
            verbose = true;
             */
        }
        PMWP(verbose > 1,
             "Reading parameters file: '%s'\n",
             ((parameters_dir + parameters_subdir + parameters_file_name).c_str()));
        /*
        if(verbose)
        {
            std::cout << "Reading parameters file: " 
                      << parameters_dir + parameters_subdir + parameters_file_name
                      << std::endl;
        }
         */
        
        if(vm.count("resume_itr") && vm.count("resume_subdir"))
        {
            resume = true;
            params.add(":experiment", "resume_iteration", resume_iteration);
            resume_path = results_dir + resume_subdir;
            params.add(":experiment", "resume_path", resume_path);
            PMWP(verbose > 1,
                 "Resume iteration %d results from %s\n",
                 (resume_iteration)(resume_path.c_str()));
            /*
            if (verbose)
                std::cerr << "Resume iteration " << resume_iteration << " results from " << resume_path << std::endl;
             */
        }
        
        // read parameters file
        // read now, so program options can override
        
        if (resume && !vm.count("parameters"))
        {
            resume_with_old_parameter = true;
            params.read(resume_path + "/" + "parameters.config");
        }
        else
        {
            params.read(parameters_dir + parameters_subdir + "/" + parameters_file_name);
            
            if( vm.count("parameters") )
            {
                params.add(":experiment","parameters_file_name",parameters_file_name);
            }
            if( vm.count("parameters_dir") )
            {
                params.add(":environment","parameters_dir",parameters_dir);
            }
            if( vm.count("parameters_subdir") )
            {
                params.add(":experiment","parameters_subdir",parameters_subdir);
            }
            
            if( vm.count("seed") )
            {
                params.add(":experiment","random_seed",random_seed);
            }
            
            if( vm.count("generate") )
            {
                generate = true;
                if( vm.count("sequence") )
                {
                    params.add(":generate","sequence_length",generate_sequence_length);
                } else {
                    generate_sequence_length =
                    params.get_param_as<size_t>(":generate", "sequence_length");
                }
                if( generate_sequence_length == 0 )
                {
                    std::cout << "ERROR: Sequence length (-s) must be > 0" << std::endl
                    << "STOPPING" << std::endl;
                    assert(false);
                }
                if( vm.count("observable_dimension") )
                {
                    params.add(":generate","observables_dimension",generate_observables_dimension);
                } else {
                    generate_observables_dimension =
                    params.get_param_as<size_t>(":generate", "observables_dimension");
                }
                if( generate_observables_dimension == 0 )
                {
                    std::cout << "ERROR: Observable dimension (-k) must be > 0" << std::endl
                    << "STOPPING" << std::endl;
                    assert(false);
                }
                if(verbose > 1)
                {
                    std::cout << "Generating new sequence of length "
                    << generate_sequence_length << " and dimension "
                    << generate_observables_dimension
                    << " using parameters: '"
                    << parameters_file_name
                    << "'" << std::endl;
                }
            }
            
            if( vm.count("data") )         // base name of data (subdirectory name and base for data files)
            {
                params.add(":experiment","data_file_name",data_file_name);
            }
            
            if( vm.count("timestamp") )    // only used during generation
            {
                if( !data_timestamp.empty() )
                {
                    params.add(":experiment","data_timestamp",data_timestamp);
                }
            }
            if( vm.count("results_timestamp") )    // only used during inference
            {
                if( !results_timestamp.empty() )
                {
                    params.add(":experiment","results_timestamp",results_timestamp);
                }
            }
            if( vm.count("data_subdir") )  // only used during inference
            {
                PMWP(verbose > 1,
                     "Reading data from data subdirectory: '%s'\n",
                     (data_subdir.c_str()));
                /*
                if(verbose)
                {
                    std::cout << "Reading data from data subdirectory: '"
                    << data_subdir << "'" << std::endl;
                }
                 */
                params.add(":experiment","data_subdir",data_subdir);
            }
            if( vm.count("data_dir") )     // root of all data subdirectories
            {
                params.add(":environment","data_dir",data_dir);
            }
            
            if( vm.count("results") )
            {
                params.add(":experiment","results_file_name",results_file_name);
            }
            if( vm.count("results_dir") )
            {
                params.add(":environment","results_dir",results_dir);
            }
            if( vm.count("iterations") )
            {
                params.add(":experiment","iterations",iterations);
            }
            
            if( vm.count("weights_file") )  // manually-specified .weights file
            {
                params.add("Known_weights","weights_file",weights_file);
            }
            
        }
    }
    catch(po::error & e)
    {
        std::cerr << "Program options ERROR: " << e.what() << std::endl << std::endl;
        std::cerr << odesc << std::endl;
        return ERROR_IN_COMMAND_LINE;
    }

    // Set random_seed_value
    // update based on read params or prog options
    if (!resume_with_old_parameter)
    {
        random_seed = params.get_param_as<std::string>(":experiment","random_seed");
        if( random_seed.compare(NO_RANDOM_SEED) == 0 )
        {
            params.add(":experiment","random_seed_value", get_usec_random_seed()); // kjb::DEFAULT_SEED;
        } else {
            params.add(":experiment","random_seed_value",random_seed);
        }
        
        data_dir = params.get_param_as<std::string>(":environment", "data_dir");
        results_dir = params.get_param_as<std::string>(":environment", "results_dir");
        ensure_directory_exists(data_dir, verbose);
        parameters_dir = params.get_param_as<std::string>(":environment", "parameters_dir");
        parameters_subdir = params.get_param_as<std::string>(":experiment", "parameters_subdir");
        params.add(":experiment", "params_path", parameters_dir + parameters_subdir + "/");
    }

    // create or read from data directory and timestamp parameter
    if(generate)
    {
        data_file_name = params.get_param_as<std::string>(":experiment", "data_file_name");
        
        // This is the path to use for all output of generated data
        // this parameter DOES get re-used; we separate the timestamped-dir from the data_file_name
        if (!params.exists(":experiment", "data_timestamp"))
        {
            std::string timestamp = get_timestamp_string();
            params.add(":experiment","data_timestamp",timestamp);
            data_subdir = data_file_name + "_" + timestamp + "/";
        } else {
            data_timestamp = params.get_param_as<std::string>(":experiment", "data_timestamp");
            data_subdir = data_file_name + "_" + data_timestamp + "/";
        }
        params.add(":experiment","data_subdir", data_subdir);
        data_path = data_dir + data_subdir;
        params.add(":experiment","data_path",data_path);
        ensure_directory_exists(data_path, verbose);
        // save generating parameters to file
        params.save(data_path + "parameters.config");
    } else {
        //if (resume_with_old_parameter)
        //{
          //  results_path = params.get_param_as<std::string>(":experiment", "results_path");
          //  params.add(":experiment", "data_path", results_path);
        //}
        //else
        if (!resume_with_old_parameter)
        {
            data_subdir = params.get_param_as<std::string>(":experiment","data_subdir") + "/";
            data_path = data_dir + data_subdir;
            params.add(":experiment","data_path",data_path);
            results_dir = params.get_param_as<std::string>(":environment","results_dir");
            results_file_name = params.get_param_as<std::string>(":experiment","results_file_name");
            if(!params.exists(":experiment", "results_timestamp"))
            {
                results_timestamp = get_timestamp_string();
                params.add(":experiment", "results_timestamp", results_timestamp);
            } else {
                results_timestamp = params.get_param_as<std::string>(":experiment", "results_timestamp");
            }
            
            // CTM 20151106: results_timestamp now plays role of experiment replication number
            results_subdir = results_file_name + "/" + results_timestamp + "/";
            // results_subdir = results_file_name + "_" + results_timestamp + "/";
            
            params.add(":experiment", "results_subdir", results_subdir);
            results_path = results_dir + results_subdir;
            //// create experiment directory
            ensure_directory_exists(results_dir, verbose);
            ensure_directory_exists(results_path, verbose);
            // this parameter does not get re-used; just for associating results with generated parameters
            params.add(":experiment","results_path",results_path);
            params.save(results_path + "parameters.config");
            // save experiment parameters to file
        }
    }
    return SUCCESS;
}

