# Hamlet experiments

## Status

CTM 20160619: New hamlet directory structure breaks run_experiment, need to finish refactoring



# New Hamlet directory structure:


    projects/hamlet/

      <symlink-to-executable>  # venti (or local)

      data/    # archived      # venti
        data/
        results/
        plots/            # for published plots

      hamlet_experiment/  # ml4ai git repo
        parameters/
        plots/            # temporary
        results/          # temporary
        scripts/
          python/
          r/
             queries/
             scripts/

      src/                # kjb/projects root
        hdp_hmm_lt/
          <executable_context>