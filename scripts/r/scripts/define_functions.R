require(abind)
require(ggplot2)

try_setwd <- function(dir)
{
    tryCatch({
        setwd(dir)
        }, error = function(e) {
            stop(paste("Can't change directory to", dir))
        })
}

get_directories <- function(project_root)
{
    results_root = paste(project_root, "results/", sep = "")
    data_root = paste(project_root, "data/", sep = "")
    visualization_root = paste(project_root, "fig/", sep = "")
    return(list(results_root = results_root, data = data_root, fig_root= visualization_root))
}

assign_colors <- function(model)
{
  colors <- c("#FA8072", "#00CED1", "#3CB371", "#FFA500", "#9400D3") 
  if (model == "noLT") return(colors[1])
  else if (model == "LT") return(colors[2])
  else if (model == "BFact") return(colors[3])
  else if (model == "Sticky") return(colors[4])
  else if (model == "StickyLT") return(colors[5])
}

fill_specs <- function(specs)
{
  if (is.null(specs$labels))
  {
    lab <- NULL
    color <- NULL
    for (i in 1:nrow(specs))
    {
      lab[i] <- strsplit(as.character(specs$model[i]),"_")[[1]][1]
      color[i] <- assign_colors(lab[i])
    }
    specs$labels <- lab
    specs$colors <- color 
  }
  return(specs)
}

## import the data frame containing data locations
get_specs <- function(query_file, results_dir, data_set, comparison_name)
{
    specs <-
        read.table(
            paste("../queries/", query_file, sep = ""),
            header = TRUE
        )
    specs <- subset(specs, comparison == comparison_name)
    specs$colors <- paste("#",specs$colors, sep="")
    specs <- fill_specs(specs)
    return(
        list(models = as.character(specs$model),
             results = results_dir,
             dataset = data_set,
             comparison = comparison_name,
             df = specs))
}

## create a list of data frames collected from the directories
## listed in specs$specs_var_name with filenames `output_type`

get_scalar_or_vector_data <- function(specs, output_type, paths, max_iteration)
{
    models <- specs$models
    data_list <- rep(list(list()), length(models))
    names(data_list) <- unique(models)
    for(m in models)
    {
        cur_path <- getwd()
        model_dir <-
            paste(
                paths$results, "/",
                specs$results, "/",
                specs$dataset, "/",
                m, "/", sep = "")
        #print(model_dir)
        try_setwd(model_dir)
        items <- Sys.glob("*")
        try_setwd(cur_path)
        for(i in items)
        {
            print(paste('Inputting data from ', model_dir, "/",
                        i, "/",
                        output_type, ".txt",
                        sep = ""))
            next_data <-
                subset(read.table(
                    paste(
                        model_dir, "/",
                        i, "/",
                        output_type, ".txt",
                        sep = ""),
                    header = FALSE,
                    skip = 1
                ), V1 <= max_iteration)
            data_list[[m]] <-
                append(data_list[[m]], list(next_data))
        }
    }
    return(data_list)
}

get_matrix_data <- function(specs, output_type, paths)
{
    groups <- specs$models
    data_list <- rep(list(list()), length(groups))
    names(data_list) <- unique(groups)
    for (m in groups)
    {
      cur_path <- getwd()
      model_dir <-
        paste(
          paths$results, "/",
          specs$results, "/",
          specs$dataset, "/",
          m, "/", sep = "")
      try_setwd(model_dir)
      items <- Sys.glob("*")
      try_setwd(cur_path)
      for (i in items)
      {
        data_path <-
          paste(
            model_dir, "/",
            i, "/",
            output_type, "/",
            sep="")
        file_names <- dir(data_path)
        first_matrix <-
          as.matrix(
            read.table(
              paste(data_path, file_names[1], sep = ""),
              header = FALSE))
        r <- nrow(first_matrix)
        c <- ncol(first_matrix)
        matrix_stack <- array(0, dim = c(r,c,length(file_names)))
        for(t in 1:length(file_names))
        {
          print(paste('Inputting matrix from', data_path, file_names[t], sep = ""))
          next_matrix <-
            as.matrix(
              read.table(
                paste(data_path, file_names[t], sep = ""),
                header = FALSE))
          matrix_stack[,,t] <- next_matrix
        }
        data_list[[m]] <-
          append(data_list[[m]], list(matrix_stack))
      }
    }
    return(data_list)

    #for(i in 1:length(specs$models))
    #{
     #   data_path <-
     #      paste(
     #           paths$results, "/",
     #           specs$results, "/",
     #           specs$dataset, "/",
     #           specs$models[i], "/",
     #           output_type, "/", sep = ""
     #           )
      #  file_names <- dir(data_path)
      #  first_matrix <-
      #      as.matrix(
      #          read.table(
      #              paste(data_path, file_names[1], sep = ""),
      #              header = FALSE))
      #  r <- nrow(first_matrix)
      #  c <- ncol(first_matrix)
      #  matrix_stack <- array(0, dim = c(r,c,length(file_names)))
      #  for(t in 1:length(file_names))
      #  {
      #      next_matrix <-
      #          as.matrix(
      #              read.table(
      #                  paste(data_path, file_names[t], sep = ""),
      #                  header = FALSE))
      #      matrix_stack[,,t] <- next_matrix
      #  }
      #  data_list[[groups[i]]] <-
      #      append(
      #          data_list[[groups[i]]],
      #          list(matrix_stack)
      #          )
#    }
#    return(data_list)
}

collect_data_as_scalar <- function(data_list, summary_function = I)
{
    result = list()
    for(l in data_list)
    {
        vals <- numeric(0)
        iterations <- l[[1]][,1]
        for(df in l)
        {
            newdata <- summary_function(df[,-1])
            length(newdata) <- length(iterations)
            vals <- cbind(vals, newdata)
        }
        l[[1]] <- l[[1]][order(l[[1]][,1]),]
        result <- append(result, list(vals))
    }
    names(result) <- names(data_list)
    return(list(iterations = iterations, values = result))
}

summarize_scalar_data_across_runs <- function(collapsed_data, smoothing_window_size)
{
    print(paste("Summarizing data with smoothing window", smoothing_window_size))
    result <- list()
    center_iteration <-
        floor(collapsed_data$iterations / smoothing_window_size) * smoothing_window_size +
            0.5 * smoothing_window_size
    for(d in collapsed_data$values)
    {
        dd <- apply(d, 2, function(x){tapply(x, center_iteration, mean)})
        result <-
            append(
                result,
                list(data.frame(
                    mean = apply(dd, 1, mean, na.rm = TRUE),
                    se_upper = apply(dd, 1, function(x) {mean(x, na.rm = TRUE) + sd(x, na.rm = TRUE) / sqrt(length(x))}),
                    se_lower = apply(dd, 1, function(x) {mean(x, na.rm = TRUE) - sd(x, na.rm = TRUE) / sqrt(length(x))}),
                    cint_upper =
                        apply(dd, 1, function(x)
                            {mean(x, na.rm = TRUE) + sqrt(var(x, na.rm = TRUE) / length(x)) * 2 * qt(0.995, length(x) - 1)}),
                    cint_lower =
                        apply(dd, 1, function(x)
                            {mean(x, na.rm = TRUE) - sqrt(var(x, na.rm = TRUE) / length(x)) * 2 * qt(0.995, length(x) - 1)}),
                    quantile_upper =
                        apply(dd, 1, function(x) {quantile(x, 0.9, na.rm=TRUE)}),
                    quantile_lower =
                        apply(dd, 1, function(x) {quantile(x, 0.1, na.rm=TRUE)}),
                    median =
                        apply(dd, 1, median, na.rm=TRUE))))

    }
    names(result) <- names(collapsed_data$values)
    return(list(iterations = unique(center_iteration), values = result))
}

summarize_matrix_data_across_iterations <-
    function(
        data_list,
        summary_function = mean,
        burnin_samples = 1
        )
{
    result <- list()
    for(l in data_list)
    {
        group_result <- list()
        for(m in l)
        {
            group_result <-
                append(
                    group_result,
                    list(apply(m[,,-c(1,burnin_samples)],
                               MARGIN = c(1,2),
                               FUN = summary_function))
                    )
        }
        matrix_result <- do.call("abind", list(group_result, along = 3))
        result <- append(result, list(matrix_result))
    }
    names(result) <- names(data_list)
    return(result)
}

summarize_matrix_data_across_iterations_and_runs <-
    function(
        data_list,
        iteration_summary_function = mean,
        run_summary_function = mean,
        burnin_samples = 1
        )
{
    print('Inputting result matrix...')
    result <- list()
    data_by_run <-
        summarize_matrix_data_across_iterations(
            data_list, iteration_summary_function, burnin_samples)
    for(l in data_by_run)
    {
        result <-
            append(
                result,
                list(apply(l, MARGIN = c(1,2), FUN = run_summary_function)))
    }
    names(result) <- names(data_list)
    return(result)
}

plot_scalar_by_iteration <-
    function(
        specs,
        output_type,
        paths,
        smoothing_window_size,
        summary_function = I,
        error_var = "cint",
        yrange = c(-Inf, Inf),
        burnin_samples = 10,
        max_iteration
        )
{
    print(paste('ploting scalar plot for ', output_type, '...', sep=""))
    output_path <- paste(paths$fig_root, specs$results, "/", specs$comparison, "/", specs$dataset, "/", sep = "")
    if(!file.exists(output_path)) dir.create(output_path, recursive = TRUE)
    #if (file.exist(paste(output_path, "/", output_type, ".pdf", sep = "")))
    #{
      #print(paste(output_path, "/", output_type, ".pdf exist!", sep = ""))
    #}
    #else
    #{
      results_list <- get_scalar_or_vector_data(specs, output_type, paths, max_iteration)
      collected_data <-
        collect_data_as_scalar(
          results_list, summary_function = summary_function
        )
      summarized_data <- summarize_scalar_data_across_runs(collected_data, smoothing_window_size)
      t <- summarized_data$iterations
      index_subset = t > burnin_samples
      ## calculate a suitable range to plot
      ## Use ggplot does not require to compute the range for plotting
      ## Put it into one data frame with factors
      groups <- specs$models
      plot_data <- data.frame()
      color_label <- specs$df
      for (g in unique(groups))
      {
        iter <- t[index_subset]
        m <- summarized_data$values[[g]]$mean[index_subset]
        lwr <- summarized_data$values[[g]][[paste(error_var,"_lower",sep = "")]][index_subset]
        upr <- summarized_data$values[[g]][[paste(error_var,"_upper",sep = "")]][index_subset]
        plot_data_each_group <- data.frame(iter, m, lwr, upr)
        plot_data_each_group$model <- color_label$labels[color_label$model==g]
        #plot_data_each_group$model <- strsplit(g, "_")[[1]][1]
        plot_data <- rbind(plot_data, plot_data_each_group)
      }
      names(plot_data) <- c("iter", "m", "lwr", "upr","model")
      group.colors <- color_label$colors
      names(group.colors) <- color_label$labels
      #group.colors <- c("#FA8072", "#00CED1", "#3CB371", "#FFA500", "#9400D3")
      #group.colors <- c("noLT" = "#FA8072", "LT" ="#00CED1", "BFact" = "#3CB371", "Sticky" = "#FFA500", "StickyLT" = "#9400D3")
      #plot_data$model <- factor(plot_data$model, levels=c('noLT','LT','BFact','Sticky','StickyLT'))
      #BFact_subset <- subset(plot_data, model=="BFact")
      #plot_data <- subset(plot_data, model!="BFact")
      #plot_data <- rbind(plot_data, BFact_subset)
      #plot_data <- plot_data[order(plot_data$model, plot_data$iter),]
      print(paste('Output to', output_path, "/", output_type, ".pdf", sep = ""))
      #pdf(paste(output_path, "/", output_type, ".pdf", sep = ""))
      ggplot(plot_data) +
        geom_ribbon(aes(x=iter, ymin=lwr, ymax=upr, group=model, fill=model), alpha=0.4) +
        geom_line(aes(x=iter, y=m, group=model, color=model)) +
        geom_point(aes(x=iter, y=m, color=model, shape=model)) +
        labs(x="Iterations", y=output_type) +
        scale_fill_manual(values=group.colors) +
        scale_color_manual(values=group.colors) +
        theme(
          axis.text=element_text(size=15),
          axis.title=element_text(size=20),
          legend.text=element_text(size=20),
          legend.title=element_text(size=24),
          aspect.ratio = 0.35
        )
      ggsave(paste(output_path, "/", output_type, ".pdf", sep = ""), width = 8, height = 2.8)
      #dev.off()
      print('done.')
    #xf}
      #lowest_val <- Inf
      #highest_val <- -Inf
      #n_iterations = max(t, na.rm = TRUE)
      #for(l in summarized_data$values)
      #{
        #lowest_val = max(min(lowest_val, min(l[[paste(error_var,"_lower", sep = "")]][index_subset], na.rm = TRUE)), yrange[1])
        #highest_val = min(max(highest_val, max(l[[paste(error_var,"_upper", sep = "")]][index_subset], na.rm = TRUE)), yrange[2])
      #}
      #pdf(paste(output_path, "/", output_type, ".pdf", sep = ""))
      #print(paste('Output to', output_path, "/", output_type, ".pdf", sep = ""))
      #plot(
        #NULL, xlim = c(0, n_iterations), ylim = c(lowest_val, highest_val),
        #xlab = "Iteration", ylab = output_type)
      #groups <- specs$models
      #plot_vars <- 1:length(unique(groups))
      #names(plot_vars) <- unique(groups)
      #for(g in unique(groups))
      #{
        #m <- summarized_data$values[[g]]$mean
        #lwr <- summarized_data$values[[g]][[paste(error_var,"_lower",sep = "")]]
        #upr <- summarized_data$values[[g]][[paste(error_var,"_upper",sep = "")]]
        #lines(t[index_subset], m[index_subset], lty = plot_vars[g])
        #lines(t[index_subset], lwr[index_subset], lty = plot_vars[g], lwd = 0.25)
        #lines(t[index_subset], upr[index_subset], lty = plot_vars[g], lwd = 0.25)
        ## arrows(x0 = t[index_subset],
        ##        y0 = lwr[index_subset],
        ##        y1 = upr[index_subset],
        ##        angle = 90, code = 3,
        ##        length = 0.1, lty = plot_vars[g])
      #}
      #legend("bottomright", lty = plot_vars, legend = unique(groups))
      #dev.off()
    #}
    #print('done.')
}

format_data_for_binary_matrix_plot <-
    function(
        specs, output_type, paths,
        iteration_summary_function = mean,
        run_summary_function = mean,
        burnin_samples = 1,
        groundtruth
        )
{
    print(paste('Inputting ground truth matrix from ',
                paths$data, "/", groundtruth, "/", "states.txt", sep = ""))
    ground_truth_data <-
        as.matrix(
            read.table(
                paste(paths$data, "/", groundtruth, "/", "states.txt", sep = "")))
    data_list <- get_matrix_data(specs, output_type, paths)
    result_matrices <-
        summarize_matrix_data_across_iterations_and_runs(
            data_list,
            iteration_summary_function, run_summary_function,
            burnin_samples)
    return(list(gt = ground_truth_data, results = result_matrices))
}

#plot_binary_matrices <- function(specs, data, paths)
#{
 #   T <- nrow(data$gt)
 #  D <- ncol(data$gt)
 #   groups <- names(data$results)
 #   G <- length(groups)
 #   pdf(
 #      paste(
 #           paths$vis, specs$results, "/grids.pdf", sep = ""))
 #   par(mfrow = c(G + 1, 1), mar = c(1,3,1,1), mgp = c(1,1,0))
 #   image(data$gt, x = seq(0.5, T + 0.5), y = seq(0.5, D + 0.5),
 #         col = gray.colors(100), xlab = "", ylab = "Ground Truth",
 #         xaxt = "n", yaxt = "n")
 #   for(g in groups)
 #   {
 #       m <- data$results[[g]]
        ## image(data$gt, x = seq(0.5, T + 0.5), y = seq(0.5, D + 0.5),
        ##       col = gray.colors(100), xlab = "", ylab = g,
        ##       xaxt = "n", yaxt = "n")
        ## image(0.5 + sign(data$gt - m) * (data$gt - m)^2 / 2, x = seq(0.5, T + 0.5), y = seq(0.5, D + 0.5),
        ##       col = gray.colors(100), xlab = "", ylab = g,
        ##       xaxt = "n", yaxt = "n")
 #       image(m, x = seq(0.5, T + 0.5), y = seq(0.5, D + 0.5),
 #             col = gray.colors(100), xlab = "", ylab = g,
 #             xaxt = "n", yaxt = "n")
 #   }
 #   dev.off()
#}

plot_binary_matrices <- function(specs, data, paths)
{
  output_dir <- paste(paths$fig_root, "/",
                      specs$results, "/",
                      specs$comparison, "/",
                      specs$dataset, "/",
                      sep="")
  if (!file.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
  T <- nrow(data$gt)
  D <- ncol(data$gt)
  print(paste('Output binary matrix to', output_dir, "/groundtruth.pdf", sep=""))
  pdf(paste(output_dir, "/groundtruth.pdf", sep=""), width=8, height=1.2)
  #old_par <- par()
  par(mar = c(1,4,1,1))
  image(data$gt, x = seq(0.5, T + 0.5), y = seq(0.5, D + 0.5),
        col = gray.colors(100), xlab="", ylab="Groundtruth",
        xaxt="n", yaxt="n", cex.lab=1)
        #par(old_par)
  dev.off()
  models <- names(data$results)
  for (ms in models)
  {
    output_path <- paste(output_dir, "/", ms, "/", sep="")
    if (!file.exists(output_path)) dir.create(output_path, recursive = TRUE)
    pdf(paste(output_path, "binary_state.pdf", sep=""), width=8, height=1.2)
    m <- data$results[[ms]]
    T <- nrow(m)
    D <- ncol(m)
    par(mar = c(1,4,1,1))
    image(m, x = seq(0.5, T + 0.5), y = seq(0.5, D + 0.5),
          col = gray.colors(100), xlab="", ylab=strsplit(ms, "_")[[1]][1],
          xaxt="n", yaxt="n", cex.lab=1)
    dev.off()
  }
  print('done.')
}

count_nonzero_entries_per_row <- function(data_matrix)
{
    return(apply(data_matrix, 1, function(x){return(sum(x != 0))}))
}

states_to_reach_one_minus_epsilon <- function(weight_vector_array, tol = 0.001)
{
    apply(
        weight_vector_array, 1,
        function(weight_vector)
        {
            sum(cumsum(sort(weight_vector, decreasing = TRUE)) < 1 - tol)
        }
        )
}

#make_key_scalar_plots <-
#    function(
#        query_file,
#        results_dir,
#        data_set,
#        burnin_samples,
#        paths,
#        comparison_name,
#        smoothing_window_size,
#        plot.vars = c("F1_score", "precision", "recall", "accuracy")
#        )
#{
#    specs <- get_specs(query_file, results_dir, data_set, comparison_name)
#    for(v in plot.vars)
#    {
#        plot_scalar_by_iteration(
#            specs, v, burnin_samples = burnin_samples, paths = paths,
#            summary_function = I,
#            smoothing_window_size)
#    }
#    if("n_dot" %in% plot.vars)
#    {
#        plot_scalar_by_iteration(
#            specs, "n_dot", burnin_samples = burnin_samples,
#            summary_function = count_nonzero_entries_per_row,
#            paths = paths,
#            smoothing_window_size)
#    }
    ## if(binary)
    ## {
    ##     binary_matrices <-
    ##         format_data_for_binary_matrix_plot(
    ##             specs, "thetastar",
    ##             burnin_samples = burnin_samples
    ##             )
    ##     plot_binary_matrices(specs, binary_matrices)
    ## }
#}

#make_scalar_plots_batch <-
#    function(
#        query_file,      #name of a text file w/ list of
#                         #leaf subdirectories
#        data_set,        #name of root directory after results_root
#        burnin_samples,  #number of logged iteration to discard as burnin
#        path_glob,       #a glob expression indicating which datasets within data_set to use
#        smoothing_window_size,
#        extra.plot.vars = c(),
#        base.plot.vars = c("F1_score", "precision", "recall",
#               "accuracy"),
#        project_root = "../../../../data/"
#        )
#{
#    specs <-
#        read.table(
#            paste("../queries/", query_file, sep = ""),
#            header = TRUE
#        )
#    comparisons <- specs$comparison
#    root <- get_directories(project_root = project_root)
#    cur_path <- getwd()
#    results_dir <- paste(root$results, "/", data_set, sep = "")
    ## print(results_dir)
#    try_setwd(results_dir)
#    paths <- Sys.glob(path_glob)
#    try_setwd(cur_path)
#    for(p in paths)
#    {
#        print(p)
#        for(comp in unique(comparisons))
#        {
#            print(paste("    ", comp, sep = ""))
#            make_key_scalar_plots(
#                query_file = query_file,
#                results_dir = data_set,
#                data_set = p,
#                burnin_samples = burnin_samples,
#                paths = root,
#                comparison_name = comp,
#                plot.vars = c(base.plot.vars, extra.plot.vars),
#                smoothing_window_size
#            )
#            print("........done.")
#        }
#    }
#}

collect.iterations <- function(path)
{
    result <- as.matrix(read.table(paste(path, "/", "thetastar/00000.txt", sep = "")))
    result <- array(the.array, c(dim(the.array), 1, 1))
    for(dd in dir(paste(path, "/thetastar", sep = ""))[-1])
    {
        new.array <- as.matrix(read.table(paste(d, "/thetastar/", dd, sep = "")))
        new.array <- array(new.array, c(dim(new.array), 1, 1))
        result <- abind(result, new.array, along = 3)
    }
    return(result)
}

create.thetastar.array <- function(root, exclusions)
{
    d <- dir(root)[1]
    subdir <- paste(root, "/", d, sep = "")
    the.array <- collect.iterations(subdir)
    ## the.array <- as.matrix(read.table(paste(d, "/", "thetastar/00000.txt", sep = "")))
    ## the.array <- array(the.array, c(dim(the.array), 1, 1))
    ## for(dd in dir(paste(d, "/thetastar", sep = ""))[-1])
    ## {
    ##     new.array <- as.matrix(read.table(paste(d, "/thetastar/", dd, sep = "")))
    ##     new.array <- array(new.array, c(dim(new.array), 1, 1))
    ##     the.array <- abind(the.array, new.array, along = 3)
    ## }
    for(d in dir(root)[-c(1,exclusions)])
    {
        ## slice.array <- as.matrix(read.table(paste(d, "/thetastar/00000.txt", sep = "")))
        ## slice.array <- array(slice.array, c(dim(slice.array), 1, 1))
        ## for(dd in dir(paste(d, "/thetastar", sep = ""))[-1])
        ## {
        ##     new.array <- as.matrix(read.table(paste(d, "/thetastar/", dd, sep = "")))
        ##     new.array <- array(new.array, c(dim(new.array), 1, 1))
        ##     slice.array <- abind(slice.array, new.array, along = 3)
        ## }
        ## the.array <- abind(the.array, slice.array, along = 4)
        subdir <- paste(root, "/", d, sep = "")
        the.array <- abind(the.array, collect.iterations(subdir), along = 4)
    }
}

plot_scalar_density_by_model <-
    function(
      specs,
      output_type,
      paths,
      #xrange = c(-Inf, Inf),
      #yrange = c(-Inf, Inf),
      summary_function = I,
      burnin_samples = 10,
      max_iteration
    )
{
      print(paste('Plot density plot for ', output_type, sep=""))
      output_path <- paste(paths$fig_root, specs$results, "/", specs$comparison, "/", specs$dataset, "/", sep = "")
      if(!file.exists(output_path)) dir.create(output_path, recursive = TRUE)
      #if (file.exist(paste(output_path, "/", output_type, "_density.pdf", sep = "")))
      #{
        #print(paste(output_path, "/", output_type, "_density.pdf exist!", sep = ""))
      #}
      #else
      #{
        results_list <- get_scalar_or_vector_data(specs, output_type, paths, max_iteration)
        collected_data <- collect_data_as_scalar(results_list, summary_function = summary_function)
        density_data <- collect_data_for_density_plot(collected_data, burnin_samples)
        groups <- specs$models
        density_plot_data <- data.frame()
        color_label <- specs$df
        for (g in unique(groups))
        {
          #print(density_data[[g]])
          plot_data <- data.frame(value = density_data[[g]])
          plot_data$model <- color_label$labels[color_label$model==g]
          #plot_data$color <- color_label$colors[color_label$model==g]
          #plot_data$model <- strsplit(g, "_")[[1]][1]
          density_plot_data <- rbind(density_plot_data, plot_data)
        }
        names(density_plot_data) <- c("value", "model")
        group.colors <- color_label$colors
        names(group.colors) <- color_label$labels
        #group.colors <- c("noLT" = "#FA8072", "LT" ="#00CED1", "BFact" = "#3CB371", "Sticky" = "#FFA500", "StickyLT" = "#9400D3")
        #BFact_subset <- subset(density_plot_data, model=="BFact")
        #density_plot_data <- subset(density_plot_data, model!="BFact")
        #density_plot_data <- rbind(density_plot_data, BFact_subset)
        #plot_data$model <- factor(plot_data$model, levels=c('noLT','LT','BFact','Sticky','StickyLT'))
        #plot_data <- plot_data[order(plot_data$model),]
        print(paste('Output density plot to', output_path, "/", output_type, "_density.pdf", sep = ""))
        width = (max(density_plot_data$value) - min(density_plot_data$value))/50
        ggplot(density_plot_data, aes(value, fill=model)) +
          geom_histogram(alpha=0.4, aes(y=..density..), position='identity', binwidth=width) +
          geom_line(stat="density", aes(color=model)) +
          labs(x=output_type, y="density") +
          scale_fill_manual(values=group.colors) +
          scale_color_manual(values=group.colors) +
          theme(
            axis.text=element_text(size=15),
            axis.title=element_text(size=20),
            legend.text=element_text(size=20),
            legend.title=element_text(size=24),
            aspect.ratio = 0.35
          )
        ggsave(paste(output_path, "/", output_type, "_density.pdf", sep = ""), width = 8, height = 2.8)


        #x_lowest_val <- Inf
        #x_highest_val <- -Inf
        #y_lowest_val <- Inf
        #y_highest_val <- -Inf
        #print(density_data)
        #for (l in density_data)
        #{
         # x_lowest_val = max(min(x_lowest_val, min(density(l, na.rm=TRUE)$x)), xrange[1])
         # x_highest_val = min(max(x_highest_val, max(density(l, na.rm=TRUE)$x)), xrange[2])
         # y_lowest_val = max(min(y_lowest_val, min(density(l, na.rm=TRUE)$y)), yrange[1])
         # y_highest_val = min(max(y_highest_val, max(density(l, na.rm=TRUE)$y)), yrange[2])
        #}
        #print(paste('Output density plot to', output_path, "/", output_type, "_density.pdf", sep = ""))
        #pdf(paste(output_path, "/", output_type, "_density.pdf", sep = ""))
        #plot(
         # NULL, xlim = c(x_lowest_val, x_highest_val), ylim=c(y_lowest_val, y_highest_val),
         # xlab = output_type, ylab = "density")
        #groups <- specs$models
        #plot_vars <- 1:length(unique(groups))
        #names(plot_vars) <- unique(groups)
        #for (g in unique(groups))
        #{
         # lines(density(density_data[[g]]), lty = plot_vars[g], col = plot_vars[g], lwd = 2)
        #}
        #labels <- NULL
        #split_group_name <- strsplit(unique(groups), "_")
        #for (i in 1:length(split_group_name))
        #{
         # labels[i] <- split_group_name[[i]][1]
        #}
        #legend("topright", lty = plot_vars, col = plot_vars, legend = labels)
        #dev.off()
      #}
      print('done.')
}

collect_data_for_density_plot <- function(collapsed_data, burnin_samples)
{
    print(paste("Burnin samples is ", burnin_samples))
    #print('collapsed data')
    #print(collapsed_data)
    result <- list()
    t <- collapsed_data$iterations
    index_subset = t > burnin_samples
    for (d in collapsed_data$values)
    {
      #print(c(data.matrix(d[index_subset,])))
      result <- append(result, list(c(data.matrix(d[index_subset,]))))
    }
    names(result) <- names(collapsed_data$values)
    return (result)
}

plot_acf_by_model_and_run <-
    function(
        specs,
        output_type,
        paths,
        max_iteration
        )
{
      print(paste("Plot acf for ", output_type, sep=""))
      results_list <- get_scalar_or_vector_data(specs, output_type, paths, max_iteration)
      output_dir <- paste(paths$fig_root, specs$results, "/", specs$comparison, "/", specs$dataset, "/", sep = "")
      if(!file.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
      models <- names(results_list)
      for (m in models)
      {
        num_runs <- length(results_list[[m]])
        for (i in 1:num_runs)
        {
          output_subdir <- paste(m,"/",formatC(i, width=2, flag="0", format="d"),"/",sep="")
          output_path <- paste(output_dir, output_subdir, sep="")
          if(!file.exists(output_path)) dir.create(output_path, recursive = TRUE)
          print(paste("Output to", output_path, "/", output_type, "_acf.pdf", sep=""))
          pdf(paste(output_path, "/", output_type, "_acf.pdf", sep=""))
          vector_data <- results_list[[m]][[i]][,-1]
          acf(vector_data, main=output_type)
          dev.off()
        }
      }
      print("done.")
}

plot_A_and_block_A <-
  function(specs, paths, block_code_path, threshold)
{
    print("Plotting A matrix...")
    output_dir <- paste(paths$fig_root, specs$results, "/", specs$comparison, "/", specs$dataset, "/", sep = "")
    if(!file.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
    models <- specs$models
    #print(models)
    for (m in models)
    {
      cur_path <- getwd()
      model_dir <- paste(paths$results, "/",
                         specs$results, "/",
                         specs$dataset, "/",
                         m, "/", sep="")
      #print(paths$results)
      #print(specs$results)
      #print(specs$dataset)
      #print(m)
      try_setwd(model_dir)
      items <- Sys.glob("*")
      try_setwd(cur_path)
      if ("BFact" %in% strsplit(m, "_")[[1]])
      {
        print('BFact model... to plot only A...')
        for (i in items)
        {
          #try_setwd(paste(model_dir, "/", i, "/", sep=""))
          #num_chains <- max(as.numeric(list.files(pattern="[0-9]")), na.rm=TRUE)
          #try_setwd(cur_path)
          model_A_dir <- paste(model_dir, "/", i, "/0/A/", sep="")
          try_setwd(model_A_dir)
          iterations <- as.numeric(substr(Sys.glob("*.txt"),1,5))
          last_iteration <- max(iterations)
          print(paste('last_iteration is ', last_iteration))
          last_iteration_file <- paste(formatC(last_iteration, width=5, flag="0", format="d"), "txt", sep=".")
          try_setwd(cur_path)
          output_path <- paste(output_dir, "/", m, "/", i, "/", sep="")
          if(!file.exists(output_path)) dir.create(output_path, recursive = TRUE)
          print(paste('Output to', output_path, "/", "A.pdf", sep=""))
          pdf(paste(output_path, "/", "A.pdf", sep=""))
          par(mfrow = c(4,4), mar = c(1,1,1,1))
          for (j in 0:15)
          {
            print(paste('Read in matrix from ', model_dir, "/", i, "/", j, "/A/", last_iteration_file, sep=""))
            A_ <- as.matrix(read.table(paste(model_dir, "/", i, "/", j, "/A/", last_iteration_file, sep="")))
            J_ <- nrow(A_)
            if (J_ == 1) image(t(A_), col=heat.colors(100), xaxt="n", yaxt="n")
            else image(t(A_)[J_:1,], col=heat.colors(100), xaxt="n", yaxt="n")
          }
          dev.off()
        }
      }
      else
      {
        print('Not BFact Model, plot both A and G...')
        for (i in items)
        {
          if (!file.exists(paste(model_dir, "/", i, "/G/", sep="")))
          {
            generate_block_diagonal_matrix(block_code_path, paste(model_dir, "/", i, "/", sep=""), threshold)
          }
          model_A_dir <- paste(model_dir, "/", i, "/A/", sep="")
          model_block_A_dir <- paste(model_dir, "/", i, "/G/block_A/", sep="")
          try_setwd(model_A_dir)
          iterations <- as.numeric(substr(Sys.glob("*.txt"),1,5))
          last_iteration <- max(iterations)
          last_iteration_file <- paste(formatC(last_iteration, width=5, flag="0", format="d"), "txt", sep=".")
          try_setwd(cur_path)
          print(paste("Read A from ", model_A_dir, last_iteration_file, sep=""))
          A_ <- as.matrix(read.table(paste(model_A_dir, last_iteration_file, sep="")))
          if (!file.exists(paste(model_block_A_dir, last_iteration_file, sep="")))
          {
            generate_block_diagonal_matrix(block_code_path, paste(model_dir, "/", i, "/", sep=""), threshold)
          }
          print(paste("Read block A from ", model_block_A_dir, last_iteration_file, sep=""))
          block_A_ <- as.matrix(read.table(paste(model_block_A_dir, last_iteration_file, sep="")))
          J_ <- nrow(A_)
          block_J_ <- nrow(block_A_)
          output_subdir <- paste(m,"/",i,"/",sep="")
          output_path <- paste(output_dir, output_subdir, sep="")
          print(paste('Output to ', output_path, sep=""))
          if(!file.exists(output_path)) dir.create(output_path, recursive = TRUE)
          pdf(paste(output_path, "/", "A.pdf", sep=""))
          if (J_ == 1) image(t(A_), col=heat.colors(100), xaxt="n", yaxt="n")
          else image(t(A_)[J_:1,], col=heat.colors(100), xaxt="n", yaxt="n")
          dev.off()
          pdf(paste(output_path, "/", "block_A.pdf", sep=""))
          if (block_J_ == 1) image(t(block_A_), col=heat.colors(100), xaxt="n", yaxt="n")
          else image(t(block_A_)[block_J_:1,], col=heat.colors(100), xaxt="n", yaxt="n")
          dev.off()
        }
      }
      }
}

generate_block_diagonal_matrix <- function(block_code_path, A_directory, threshold)
{
  print("generate_G...")
  t_command <- paste("t=", threshold, sep="")
  print(paste("python", block_code_path, A_directory, "true", t_command, sep=" "))
  system(paste("python", block_code_path, A_directory, "true", t_command, sep=" "))
}

make_plots <-
  function(
    query_file,
    data_set,
    burnin_samples,
    path_glob,
    smoothing_window_size,
    plot.vars,
    project_root,
    threshold,
    block_code_path,
    binary,
    groundtruth,
    max_iteration
    )
{
    specs <- read.table(paste("../queries/", query_file, sep=""), header = TRUE)
    comparisons <- specs$comparison
    root <- get_directories(project_root = project_root)
    cur_path <- getwd()
    results_dir <- paste(root$results, "/", data_set, sep="")
    print(results_dir)
    try_setwd(results_dir)
    paths <- Sys.glob(path_glob)
    try_setwd(cur_path)
    for (p in paths)
    {
      print(p)
      for (comp in unique(comparisons))
      {
        print(paste("    ", comp, sep = ""))
        make_key_plots(query_file = query_file,
                       results_dir = data_set,
                       data_set = p,
                       burnin_samples = burnin_samples,
                       paths = root,
                       comparison_name = comp,
                       plot.vars = plot.vars,
                       smoothing_window_size = smoothing_window_size,
                       threshold = threshold,
                       block_code_path = block_code_path,
                       binary = binary,
                       groundtruth = groundtruth,
                       max_iteration = max_iteration)
        print("...........done.")
      }
    }
}

make_key_plots <-
  function(
    query_file,
    results_dir,
    data_set,
    burnin_samples,
    paths,
    comparison_name,
    smoothing_window_size,
    plot.vars,
    threshold,
    block_code_path,
    binary,
    groundtruth,
    max_iteration
    )
{
    specs <- get_specs(query_file, results_dir, data_set, comparison_name)
    #no_density_and_acf <- c("F1_score", "precision", "recall", "accuracy", "n_dot")
    plot_A = FALSE
    if ("A" %in% plot.vars)
    {
      plot_A = TRUE
      plot.vars <- plot.vars[plot.vars != "A"]
    }
    for (v in plot.vars)
    {
      print(paste('Plotting for ', v, '...', sep=""))
      if (v == "n_dot")
      {
          plot_scalar_by_iteration(
                    specs, "n_dot", burnin_samples = burnin_samples,
                    summary_function = count_nonzero_entries_per_row,
                    paths = paths,
                    smoothing_window_size,
                    max_iteration = max_iteration)
        plot_scalar_density_by_model(specs = specs,
                                     output_type = "n_dot",
                                     paths = paths,
                                     burnin_samples = burnin_samples,
                                     summary_function = count_nonzero_entries_per_row,
                                     max_iteration = max_iteration)
      }
      else
      {
          plot_scalar_by_iteration(
                    specs, v, burnin_samples = burnin_samples, paths = paths,
                    summary_function = I,
                    smoothing_window_size,
                    max_iteration = max_iteration)
          plot_scalar_density_by_model(specs = specs,
                    output_type = v,
                    paths = paths,
                    burnin_samples = burnin_samples,
                    max_iteration = max_iteration)
          ## plot_acf_by_model_and_run(specs = specs,
          ##           output_type = v,
          ##           paths = paths,
          ##           max_iteration)
      }
    }
    #if("n_dot" %in% plot.vars)
    #    {
    #
    #    }
    if (plot_A)
    {
        plot_A_and_block_A(specs = specs,
                           paths = paths,
                           block_code_path = block_code_path,
                           threshold = threshold)
    }
    if (binary)
    {
      binary_matrices <-
                format_data_for_binary_matrix_plot(
                     specs, "thetastar", paths=paths,
                     burnin_samples = burnin_samples,
                     groundtruth = groundtruth
                     )
      plot_binary_matrices(specs, binary_matrices, paths)
    }
}


