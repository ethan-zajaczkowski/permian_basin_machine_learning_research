### ----- Functions For Training ----- ###

split_dataset = function(dataset_division, dataset, msg = FALSE, seed = NULL) {
  # Function for splitting the hdbscan dataset by specified amount
  
  # Set seed for reproducibility if provided
  if (!is.null(seed)) {
    set.seed(seed)
  }
  
  # Split by cluster
  all_clusters = unique(dataset$cluster)
  n_train_clusters = floor(dataset_division * length(all_clusters))
  
  # Divide into train and test
  train_clusters = sample(all_clusters, size = n_train_clusters)
  train_data = dataset %>% filter(cluster %in% train_clusters)
  test_data = dataset %>% filter(!cluster %in% train_clusters)
  
  # Print the number of clusters and samples
  if (msg == TRUE) {
    message("Train clusters: ", length(unique(train_data$cluster)), " (", nrow(train_data), " samples)",
            "  |  Test clusters: ", length(unique(test_data$cluster))," (", nrow(test_data), " samples)")
  }
  
  # Return training and testing datasets with metadata
  return(list(
    train_data = train_data, 
    test_data = test_data,
    train_clusters = train_clusters,
    test_clusters = setdiff(all_clusters, train_clusters)
  ))
}


training_testing_data_matrix = function(training_set, testing_set, features, target = "methane_eq") {
  # Creating training and testing datasets for XGBoost:
  
  # 1) Numeric feature matrix + label
  train_matrix = data.matrix(training_set[, features, drop = FALSE])
  train_label = as.numeric(training_set[[target]])
  
  test_matrix = data.matrix(testing_set[, features, drop = FALSE])
  test_label = as.numeric(testing_set[[target]])
  
  # 2) Build the DMatrix objects
  dtrain = xgb.DMatrix(data = train_matrix, label = train_label)
  dtest = xgb.DMatrix(data = test_matrix, label = test_label)
  
  # 3) Return both with metadata
  return(list(
    dtrain = dtrain, 
    dtest = dtest,
    n_train = nrow(train_matrix),
    n_test = nrow(test_matrix),
    features = features
  ))
}

model_parameters = function(eta = NULL,
                            max_depth = NULL,
                            subsample = NULL,
                            colsample_bytree = NULL,
                            min_child_weight = NULL,
                            gamma = NULL,
                            lambda = NULL,
                            alpha = NULL,
                            max_leaves = NULL,
                            scale_pos_weight = NULL,
                            tree_method = "auto") {
  # Function to input the parameters for the model
  
  params = list(
    objective = "reg:squarederror",
    eval_metric = "rmse",
    tree_method = tree_method
  )
  
  if (!is.null(eta))              params$eta               = eta
  if (!is.null(max_depth))        params$max_depth         = max_depth
  if (!is.null(subsample))        params$subsample         = subsample
  if (!is.null(colsample_bytree)) params$colsample_bytree  = colsample_bytree
  if (!is.null(min_child_weight)) params$min_child_weight  = min_child_weight
  if (!is.null(gamma))            params$gamma             = gamma
  if (!is.null(lambda))           params$lambda            = lambda
  if (!is.null(alpha))            params$alpha             = alpha
  if (!is.null(max_leaves))       params$max_leaves        = max_leaves
  if (!is.null(scale_pos_weight)) params$scale_pos_weight  = scale_pos_weight
  
  return(params)
}

train_xgb_model = function(dtrain, dtest = NULL, params, nrounds = 100, early_stopping_rounds = 10, print_every_n = 50, verbose = 0) {
  # Function to train the XGBoost Model
  
  if (!is.null(dtest)) {
    watchlist = list(train = dtrain, test = dtest)
  } else {
    watchlist = list(train = dtrain)
  }
  
  # Train the model
  model = xgb.train(
    params = params,
    data = dtrain,
    nrounds = nrounds,
    watchlist = watchlist,
    early_stopping_rounds = early_stopping_rounds,
    print_every_n = print_every_n,
    verbose = verbose
  )
  
  # Get predictions on training data
  train_pred = predict(model, dtrain)
  train_actual = getinfo(dtrain, "label")
  
  # Calculate train metrics
  train_mse = mean((train_actual - train_pred)^2)
  train_rmse = sqrt(train_mse)
  train_mae = mean(abs(train_actual - train_pred))
  
  # Calculate R² using 1 - (SSR/SST)
  ss_res = sum((train_actual - train_pred)^2)
  ss_tot = sum((train_actual - mean(train_actual))^2)
  train_r2 = 1 - (ss_res / ss_tot)
  
  # Get feature importance
  importance_matrix = xgb.importance(
    feature_names = colnames(dtrain), 
    model = model
  )
  
  # Return comprehensive results
  return(list(
    model = model,
    train_metrics = data.frame(
      train_mse = train_mse,
      train_rmse = train_rmse,
      train_mae = train_mae,
      train_r2 = train_r2
    ),
    importance = importance_matrix,
    best_iteration = model$best_iteration
  ))
}

test_xgb_model = function(model, dtest) {
  # Function for evaluating the model, gives mse, rmse, mae, and r2
  
  # 1) extract true labels and predictions for test set
  truth = xgboost::getinfo(dtest, "label")
  preds = predict(model, dtest)
  
  # 2) compute test regression metrics
  mse_val = mean((truth - preds)^2)
  rmse_val = sqrt(mse_val)
  mae_val = mean(abs(truth - preds))
  
  # Calculate R² using 1 - (SSR/SST)
  ss_res_test = sum((truth - preds)^2)
  ss_tot_test = sum((truth - mean(truth))^2)
  test_r2 = 1 - (ss_res_test / ss_tot_test)
  
  # 3) assemble into a data.frame
  results = data.frame(
    Metric = c("Mean squared error (MSE)", 
               "Root mean square error (RMSE)",
               "Mean absolute error (MAE)",
               "Test R2"),
    Value = c(mse_val, rmse_val, mae_val, test_r2),
    row.names = NULL,
    stringsAsFactors = FALSE
  )
  
  # Return results
  return(results)
}

evaluate_or_load = function(mode = 0, dataset, features, save_models = FALSE) {
  # Function to help tune the model
  # 
  # mode = 0 → run the grid-search and save the CSV
  # mode = 1 → just read the existing CSV
  
  csv_path = "datasets/xgboost_datasets/xgboost_hyperparameter_results.csv"
  
  if (mode == 0) {
    # Parameter grid
    
    param_grid = expand.grid(
      dataset_split = c(0.7, 0.75, 0.8),
      eta = c(0.01, 0.1, 0.3, 0.5),
      max_depth = c(3, 5, 7),
      subsample = c(0.6, 0.8, 1.0),
      colsample_bytree = c(0.6, 0.8, 1.0),
      min_child_weight = c(1, 5, 10),
      gamma = c(0, 0.1, 0.5, 1),
      lambda = c(0, 0.1, 0.5, 1, 5),
      alpha = c(0, 0.1, 0.5, 1, 5),
      stringsAsFactors = FALSE
    )
    
    # Initialize results storage
    metrics_list = vector("list", nrow(param_grid))
    best_models = list()  # Store top models if requested
    
    # Progress tracking
    total_models = nrow(param_grid)
    pb = txtProgressBar(min = 0, max = total_models, style = 3)
    start_time = Sys.time()
    
    cat(sprintf("Starting grid search with %d models...\n", total_models))
    
    for (i in seq_len(nrow(param_grid))) {
      p = param_grid[i, ]
      
      # Calculate time estimate (after first model)
      if (i > 1) {
        elapsed_time = as.numeric(difftime(Sys.time(), start_time, units = "secs"))
        avg_time_per_model = elapsed_time / (i - 1)
        remaining_models = total_models - i
        estimated_remaining = remaining_models * avg_time_per_model
        
        # Format time
        eta_formatted = sprintf("%02d:%02d:%02d",
                                floor(estimated_remaining/3600),
                                floor((estimated_remaining%%3600)/60),
                                floor(estimated_remaining%%60))
        
        # Print progress with ETA
        cat(sprintf("\r%d/%d models (%.1f%%) - ETA: %s          ",
                    i, total_models, (i/total_models)*100, eta_formatted))
      } else {
        cat(sprintf("\r%d/%d models (%.1f%%)          ",
                    i, total_models, (i/total_models)*100))
      }
      
      # Split data
      splits = split_dataset(p$dataset_split, dataset, msg = FALSE, seed = 123)
      train_data = splits$train_data
      test_data = splits$test_data
      
      # Create DMatrix objects
      dsplits = training_testing_data_matrix(train_data, test_data, features)
      dtrain = dsplits$dtrain
      dtest = dsplits$dtest
      
      # Set parameters
      params = model_parameters(
        eta = p$eta,
        max_depth = p$max_depth,
        subsample = p$subsample,
        colsample_bytree = p$colsample_bytree,
        min_child_weight = p$min_child_weight,
        gamma = p$gamma,
        lambda = p$lambda,
        alpha = p$alpha
      )
      
      # Train model with validation set
      result = train_xgb_model(
        dtrain,
        dtest = dtest,  # Include test set for early stopping
        params = params,
        nrounds = 500,  # Increased for early stopping
        early_stopping_rounds = 20,
        verbose = 0
      )
      
      model = result$model
      train_metrics = result$train_metrics
      
      # Test model
      test_metrics = test_xgb_model(model, dtest)
      
      # Combine all metrics
      combined_metrics = cbind(
        p,  # hyperparameters
        train_metrics,  # training metrics
        test_metrics %>%
          pivot_wider(names_from = Metric, values_from = Value),
        best_iteration = result$best_iteration,
        model_id = i
      )
      
      metrics_list[[i]] = combined_metrics
      
      # Store model if it's in top 10 by R2 so far
      if (save_models && i > 10) {
        current_results = dplyr::bind_rows(metrics_list[1:i])
        if (combined_metrics$`Test R2` >= sort(current_results$`Test R2`, decreasing = TRUE)[10]) {
          best_models[[as.character(i)]] = list(
            model = model,
            params = p,
            importance = result$importance,
            metrics = combined_metrics
          )
        }
      }
      
      setTxtProgressBar(pb, i)
    }
    
    # Print completion message
    total_time = as.numeric(difftime(Sys.time(), start_time, units = "secs"))
    time_formatted = sprintf("%02d:%02d:%02d",
                             floor(total_time/3600),
                             floor((total_time%%3600)/60),
                             floor(total_time%%60))
    
    cat(sprintf("\n\nCompleted: %d/%d models (100.0%%) in %s\n",
                total_models, total_models, time_formatted))
    close(pb)
    
    # Combine all results
    all_results = dplyr::bind_rows(metrics_list)
    
    # Save results
    write.csv(all_results, file = csv_path, row.names = FALSE)
    cat(sprintf("Results saved to: %s\n", csv_path))
    
    # Save top 10 models
    if (save_models) {
      top_10_idx = order(all_results$`Test R2`, decreasing = TRUE)[1:10]
      final_best_models = best_models[as.character(all_results$model_id[top_10_idx])]
      saveRDS(final_best_models, file = models_path)
      cat(sprintf("Top 10 models saved to: %s\n", models_path))
    }
    
    return(all_results)
    
  } else if (mode == 1) {
    # Load existing results
    results = read.csv(csv_path, stringsAsFactors = FALSE)
    cat(sprintf("Loaded %d results from: %s\n", nrow(results), csv_path))
    
    return(results)
  } else {
    stop("`mode` must be 0 (evaluate) or 1 (load).")
  }
}

### ----- Functions For Evaluation ----- ###

create_xgb_model_and_predictions = function(all_results, 
                                            dataset_split_value,
                                            eta_value,
                                            max_depth_value, 
                                            subsample_value,
                                            colsample_bytree_value,
                                            min_child_weight_value,
                                            gamma_value,
                                            lambda_value,
                                            alpha_value,
                                            observed_only_dataset,
                                            raw_dataset,
                                            features,
                                            nrounds = 100) {
  
  # Fetching Model Related Metrics
  filtered_row = subset(all_results, dataset_split == dataset_split_value & 
                          eta == eta_value & max_depth == max_depth_value & 
                          subsample == subsample_value & colsample_bytree == colsample_bytree_value &
                          min_child_weight == min_child_weight_value & gamma == gamma_value &
                          lambda == lambda_value & alpha == alpha_value)
  
  # Creating Model With Specific Parameters
  params = model_parameters(eta = eta_value, max_depth = max_depth_value, subsample = subsample_value, 
                            colsample_bytree = colsample_bytree_value, min_child_weight = min_child_weight_value, 
                            gamma = gamma_value, lambda = lambda_value, alpha = alpha_value)
  
  splits = split_dataset(dataset_split_value, observed_only_dataset, msg = FALSE)
  train_data = splits$train_data
  test_data = splits$test_data
  dsplits = training_testing_data_matrix(train_data, test_data, features)
  dtrain = dsplits$dtrain
  dtest = dsplits$dtest
  results = train_xgb_model(dtrain = dtrain, params = params, nrounds = nrounds, verbose = 0)
  model = results$model
  
  # Final Dataframe Modifications
  model_dataset = raw_dataset
  
  # Creating Predicted Column
  missing_indicator_col = which(colnames(model_dataset) == "methane_eq_missingness_indicator")
  dataset_matrix = xgb.DMatrix(data = as.matrix(model_dataset[, features]))
  predictions = predict(model, dataset_matrix)
  
  # Inserting Predicted Column
  model_dataset = model_dataset[, 1:missing_indicator_col] %>%
    bind_cols(pred_methane_eq = predictions) %>%
    bind_cols(model_dataset[, (missing_indicator_col + 1):ncol(model_dataset)])
  
  # Creating Imputing Column
  pred_methane_eq_col = which(colnames(model_dataset) == "pred_methane_eq")
  imputed_methane_eq = ifelse(
    model_dataset$methane_eq_missingness_indicator == 1,
    model_dataset$pred_methane_eq,
    model_dataset$methane_eq
  )
  
  # Inserting Imputed Column
  model_dataset = model_dataset[, 1:pred_methane_eq_col] %>%
    bind_cols(imputed_methane_eq = imputed_methane_eq) %>%
    bind_cols(model_dataset[, (pred_methane_eq_col + 1):ncol(model_dataset)])
  
  # Return results
  return(list(
    model = model,
    model_dataset = model_dataset,
    filtered_row = filtered_row,
    train_data = train_data,
    test_data = test_data,
    predictions = predictions
  ))
}

plot_observed_variable = function(observed_only_dataset, raw_dataset, title = "Original Methane EQ Distribution") {
  # Function to obtain the original observed Methane_EQ distribution
  
  # Variable
  variable = "methane_eq"
  var_sym = sym(variable)
  x_obs = observed_only_dataset[[variable]]
  
  # 2 SD window
  m_obs = mean(x_obs, na.rm=TRUE)
  s_obs = sd(x_obs, na.rm=TRUE)
  lims_obs = c(0, 0.3)
  
  # Calculate offset for text positioning
  x_range = lims_obs[2] - lims_obs[1]
  offset = x_range * 0.07  # 5% of the visible range
  
  # Mean Label
  m_obs_lbl = round(m_obs, 3)
  
  # Build Base Plot
  p_obs_base = ggplot(raw_dataset, aes(x=.data[[variable]])) +
    geom_histogram(bins=10000, fill="grey80", color="black") +
    coord_cartesian(xlim=lims_obs, ylim=c(0, 225)) +
    geom_vline(xintercept=m_obs, linetype="dotted", color="red") +
    labs(
      title="Histogram Of Target VIIRS Methane Equivalent Fuel Flaring Volumes",
      subtitle="Red = mean",
      x="Methane Equivalent Fuel Flaring Volume, Million Standard Cubic Meters / Day",
      y="Frequency"
    ) +
    theme_minimal() + 
    theme(plot.title = element_text(hjust = 0.5))
  
  # Find Y position
  pb = ggplot_build(p_obs_base)$data[[1]]
  y_top = 220  # Use a value within the visible range (below 225)
  
  # Add Mean Label
  p_obs = p_obs_base +
    annotate("text", x=m_obs + offset, y=y_top, label=paste0("Mean = ", m_obs_lbl), color="red", size=3, hjust=0)
  
  # Render
  ggplotly(p_obs, tooltip="x")
}

plot_observed_histogram = function(data, title = "Histogram Of Predicted Methane Equivalent Fuel Flaring Volumes By Model") {
  ## Function to plot the histogram of the new model but only on observed methane_eq values
  
  observed_data = data %>%
    filter(methane_eq_missingness_indicator == 0)
  
  # Variable
  variable = "pred_methane_eq"
  var_sym = sym(variable)
  x_obs = observed_data[[variable]]
  
  # 2 SD Window
  m_obs = mean(x_obs, na.rm=TRUE)
  s_obs = sd(x_obs, na.rm=TRUE)
  lims_obs = c(0, 0.3)
  
  # Calculate offset for text positioning
  x_range = lims_obs[2] - lims_obs[1]
  offset = x_range * 0.07  # 5% of the visible range
  
  # Mean Label
  m_obs_lbl = round(m_obs, 3)
  
  # Build Base Plot
  p_obs_base = ggplot(observed_data, aes(x=.data[[variable]])) +
    geom_histogram(bins=10000, fill="lightblue", color="black") +
    coord_cartesian(xlim=lims_obs, ylim=c(0, 225)) +
    geom_vline(xintercept=m_obs, linetype="dotted", color="red") +
    labs(
      title=title,
      subtitle="Red = mean",
      x="Methane Equivalent Fuel Flaring Volume, Million Standard Cubic Meters / Day",
      y="Frequency"
    ) +
    theme_minimal() + 
    theme(plot.title = element_text(hjust = 0.5))
  
  # Find Y Position
  pb = ggplot_build(p_obs_base)$data[[1]]
  y_top = 220  # Use a value within the visible range (below 225)
  
  # Add Mean Label
  p_obs = p_obs_base +
    annotate("text", x=m_obs + offset, y=y_top, label=paste0("Mean = ", m_obs_lbl), color="red", size=3, hjust=0)
  
  # Render
  ggplotly(p_obs, tooltip="x")
}

plot_model_histogram = function(data, title = "Histogram of Observed Methane_EQ (missingness_indicator = 0 & 1)") {
  # Function to plot the new XGBoost Model but on both observed and missing values combined
  
  observed_data = data
  
  # Variable
  variable = "pred_methane_eq"
  var_sym = sym(variable)
  x_obs = observed_data[[variable]]
  
  # 2 SD window
  m_obs = mean(x_obs, na.rm=TRUE)
  s_obs = sd(x_obs, na.rm=TRUE)
  lims_obs = c(max(0, m_obs-2*s_obs), m_obs+2*s_obs)
  
  # Quartiles and labels
  qs_obs = quantile(x_obs, probs=c(0.25,0.5,0.75), na.rm=TRUE)
  m_obs_lbl = round(m_obs, 3)
  q_lbls = round(qs_obs, 3)
  
  # Build base plot
  p_obs_base = ggplot(observed_data, aes(x=.data[[variable]])) +
    geom_histogram(bins=10000, fill="lightblue", color="black") +
    coord_cartesian(xlim=lims_obs) +
    geom_vline(xintercept=m_obs, linetype="dotted", color="red") +
    geom_vline(xintercept=qs_obs, linetype="dotted", color="blue") +
    labs(
      title=title,
      subtitle="Red = mean; Blue = quartiles (25%, 50%, 75%)",
      x="methane_eq",
      y="Count"
    ) +
    theme_minimal()
  
  # Find Y Positions And Add Labels
  pb = ggplot_build(p_obs_base)$data[[1]]
  y_top = max(pb$count, na.rm=TRUE)
  y_positions = y_top * c(1.00, 0.90, 0.80, 0.70)
  
  p_obs = p_obs_base +
    annotate("text", x=m_obs,     y=y_positions[1], label=paste0("Mean = ", m_obs_lbl), color="red",  size=3, hjust=0.5) +
    annotate("text", x=qs_obs[1], y=y_positions[2], label=paste0("25th Q = ", q_lbls[1]), color="blue", size=3, hjust=0.5) +
    annotate("text", x=qs_obs[2], y=y_positions[3], label=paste0("50th Q = ", q_lbls[2]), color="blue", size=3, hjust=0.5) +
    annotate("text", x=qs_obs[3], y=y_positions[4], label=paste0("75th Q = ", q_lbls[3]), color="blue", size=3, hjust=0.5)
  
  ggplotly(p_obs, tooltip="x")
}

plot_imputed_histogram = function(data, title = "Imputed imputed_methane_eq (stacked: green=orig, blue=imputed)") {
  # Function to plot the Imputed Model
  
  variable = "imputed_methane_eq"
  flag = data$methane_eq_missingness_indicator
  
  # 2 SD window and limit x-axis to 0.5
  x_imp = data[[variable]]
  m_imp = mean(x_imp, na.rm=TRUE)
  s_imp = sd(x_imp, na.rm=TRUE)
  lims_imp = c(0, 0.3)  # Fixed x-axis limits from 0 to 0.5
  
  # Calculate offset for text positioning
  x_range = lims_imp[2] - lims_imp[1]
  offset = x_range * 0.1  # 7% of the visible range
  
  # Mean Label
  m_lbl = round(m_imp, 3)
  
  # Calculate peaks (modes) for each group
  observed_data = x_imp[flag == 0]  # Green (observed)
  imputed_data = x_imp[flag == 1]   # Blue (imputed)
  
  # Function to find peak (mode) using density estimation
  find_peak <- function(data) {
    if(length(data) > 1) {
      d <- density(data, na.rm = TRUE)
      peak <- d$x[which.max(d$y)]
      return(peak)
    } else {
      return(NA)
    }
  }
  
  # Calculate peaks
  peak_observed = find_peak(observed_data)
  peak_imputed = find_peak(imputed_data)
  peak_difference = peak_imputed - peak_observed
  
  # Round for display
  peak_obs_lbl = round(peak_observed, 3)
  peak_imp_lbl = round(peak_imputed, 3)
  peak_diff_lbl = round(peak_difference, 3)
  
  # Create a new column with descriptive labels
  data_with_labels = data %>%
    mutate(data_type = ifelse(methane_eq_missingness_indicator == 0, 
                              "Original VIIRS Methane Eq.", 
                              "Imputed Methane Eq."))
  
  # Build stacked histogram using the new labels
  p_imp_base = ggplot(data_with_labels,
                      aes(x = .data[[variable]], fill = data_type)) +
    geom_histogram(bins = 10000, position = "stack", color = NA) +
    coord_cartesian(xlim = lims_imp) +
    scale_fill_manual(values = c("Original VIIRS Methane Eq."="green", "Imputed Methane Eq."="blue"), 
                      name = NULL) +  # Remove legend title
    geom_vline(xintercept = m_imp,  linetype="dotted", color="red") +
    labs(title = title, 
         x = "Methane Equivalent Fuel Flaring Volume (mmscm/d)", 
         y = "Frequency") +
    theme_minimal() +
    theme(
      legend.position = c(0.75, 0.75),  # Move inside the plot area
      legend.background = element_blank(),  # Remove background
      legend.box.background = element_blank(),  # Remove box background
      legend.key = element_rect(color = NA),  # Remove key outlines
      legend.margin = margin(0, 0, 0, 0)  # Remove margins
    )
  
  # compute label y–positions
  pb = ggplot_build(p_imp_base)$data[[1]]
  y_top = max(pb$count, na.rm=TRUE)
  
  # add mean label with offset to move it right
  p_imp_final = p_imp_base +
    annotate("text", x=m_imp + offset, y=y_top, label=paste0("Mean = ", m_lbl), color="red", size=3, hjust=0)
  
  # Print the peak values to console
  cat("Peak Analysis:\n")
  cat("Observed Peak:", peak_obs_lbl, "\n")
  cat("Imputed Peak:", peak_imp_lbl, "\n")
  cat("Peak Difference (Imputed - Observed):", peak_diff_lbl, "\n")
  
  # Create the interactive plot
  plot_output = ggplotly(p_imp_final, tooltip="x,fill")
  
  # Return both plot and peak values
  return(list(
    plot = plot_output,
    peak_observed = peak_obs_lbl,
    peak_imputed = peak_imp_lbl,
    peak_difference = peak_diff_lbl
  ))
}

plot_overlapped_histograms = function(data, title = "Overlapped Histograms of methane_eq") {
  # Function to compare the plot between true observed methane_eq values, predicted methane_eq values, and imputed methane_eq values.
  
  # 1) Compute ±2 SD window across all three
  x_obs = data$methane_eq
  x_pred = data$pred_methane_eq
  x_imp = data$imputed_methane_eq
  ms = c(mean(x_obs, na.rm=TRUE), mean(x_pred, na.rm=TRUE), mean(x_imp, na.rm=TRUE))
  ss = c(sd(x_obs,   na.rm=TRUE), sd(x_pred,   na.rm=TRUE), sd(x_imp,   na.rm=TRUE))
  lims_combined = c(max(0, min(ms - 2*ss)), max(ms + 2*ss))
  
  # 2) Pivot to long
  long_df = data %>%
    transmute(Observed = methane_eq, Predicted = pred_methane_eq, Imputed = imputed_methane_eq) %>%
    pivot_longer(cols = everything(), names_to = "Source", values_to = "Value")
  
  # 3) Build ggplot
  p = ggplot(long_df, aes(x = Value, fill = Source)) +
    geom_histogram(position = "identity", alpha = 0.4, bins = 10000, boundary = lims_combined[1], closed = "right") +
    coord_cartesian(xlim = lims_combined) +
    scale_fill_manual(values = c("Observed"  = "green","Predicted" = "blue", "Imputed" = "red")) +
    labs(title = title, x = "Value", y = "Count", fill = "Source") +
    theme_minimal()
  
  # 4) Convert to plotly and enable "click to isolate" behavior
  ggplotly(p, tooltip = "x") %>%
    layout(legend = list(itemclick  = "toggleothers", title = list(text = "<b>Click to isolate</b>")))
}

plot_feature_importance = function(model, feature_names, top_n = 20, title = "Feature Importance") {
  # Function to plot feature importance
  
  importance_matrix = xgb.importance(feature_names = feature_names, model = model)
  xgb.plot.importance(importance_matrix, top_n = top_n, main = title)
  
  invisible(importance_matrix)
}

plot_observed_vs_predicted = function(data, title = "Observed vs Predicted Methane_eq") {
  # Function to compare the original labels of methane_eq against the predicted
  
  actual_col = "methane_eq"
  predicted_col = "pred_methane_eq"
  x_label = "Predicted Methane Eq. (mscm/d)"
  y_label = "Observed Methane Eq. (mscm/d)"
  
  # Ensure data is a dataframe
  dataset_df = as.data.frame(data)
  
  # Create the plot
  plot = ggplot(dataset_df, aes_string(x = predicted_col, y = actual_col)) +
    geom_point(alpha = 0.5) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    geom_smooth(method = "lm", se = FALSE, color = "blue") +
    labs(
      title = title,
      x = x_label,
      y = y_label
    ) +
    theme_minimal()
  
  return(plot)
}

create_diagnostic_plots = function(data,main_title = "Model Diagnostic Plots") {
  ## Function for generating diagonstic plots of the model
  
  actual_col = "methane_eq"
  predicted_col = "pred_methane_eq"
  missingness_col = "methane_eq_missingness_indicator"
  
  # Filter to observed data only
  filtered_dataset = data %>%
    filter(!!sym(missingness_col) == 0)
  
  # Calculate residuals
  residuals = filtered_dataset[[actual_col]] - filtered_dataset[[predicted_col]]
  
  # Set up the plotting layout
  par(mfrow = c(2, 2))
  
  # 1. Residuals vs Predicted Values
  plot(filtered_dataset[[predicted_col]], residuals, 
       main = "Residuals vs Predicted", 
       xlab = "Fitted Values (Predictions)", 
       ylab = "Residuals")
  abline(h = 0, col = "red")
  
  # 2. Histogram of Residuals (limited to 3 SD)
  mean_resid = mean(residuals, na.rm = TRUE)
  sd_resid = sd(residuals, na.rm = TRUE)
  lower_limit = mean_resid - 3 * sd_resid
  upper_limit = mean_resid + 3 * sd_resid
  hist(residuals, 
       main = "Distribution of Residuals", 
       xlab = "Residuals", 
       breaks = 500, 
       xlim = c(lower_limit, upper_limit))
  
  # 3. Q-Q Plot of Residuals
  qqnorm(residuals, main = "Q-Q Plot of Residuals")
  qqline(residuals, col = "red")
  
  # 4. Scale-Location Plot
  standardized_residuals = residuals / sd(residuals, na.rm = TRUE)
  plot(filtered_dataset[[predicted_col]], sqrt(abs(standardized_residuals)), 
       main = "Scale-Location", 
       xlab = "Fitted Values", 
       ylab = "√|Standardized Residuals|")
  lines(lowess(filtered_dataset[[predicted_col]], sqrt(abs(standardized_residuals))), col = "blue")
  
  # Reset plotting layout
  par(mfrow = c(1, 1))
  
  # Return summary statistics invisibly
  invisible(list(
    n_observations = nrow(filtered_dataset),
    mean_residual = mean_resid,
    sd_residual = sd_resid,
    residuals = residuals
  ))
}

calculate_and_display_bias_metrics = function(data, caption = "Bias Summary Metrics") {
  # Function for evaluating and displaying bias metrics of mean bias estimate, mean absolute error, root mean square error, percent bias
  
  actual_col = "methane_eq"
  predicted_col = "pred_methane_eq"
  
  # Calculate residuals
  residuals = data[[actual_col]] - data[[predicted_col]]
  
  bias_df = data.frame(
    mean_bias_estimate = mean(residuals, na.rm = TRUE),
    mean_absolute_error = mean(abs(residuals), na.rm = TRUE),
    root_mean_square_error = sqrt(mean(residuals^2, na.rm = TRUE)),
    percent_bias = 100 * sum(residuals, na.rm = TRUE) / sum(data[[actual_col]], na.rm = TRUE)
  )
  
  colnames(bias_df) = c("Mean Bias Estimate", "MAE", "RMSE", "Percent Bias")
  
  # Create and return formatted table (fixed: use bias_df instead of bias_metrics)
  kable(bias_df, caption = caption, digits = 6, row.names = FALSE) %>%
    kable_styling(bootstrap_options = c("striped", "hover"))
}

display_train_test_metrics = function(filtered_row, caption = "Training vs Testing Metrics") {
  # Function to extract all training and testing metrics for the model
  
  # Extract metrics from filtered row
  train_r2 = filtered_row[["train_r2"]]
  test_r2 = filtered_row[["Test.R2"]]
  train_rmse = filtered_row[["train_rmse"]]
  test_rmse = filtered_row[["Root.mean.square.error..RMSE."]]
  train_mae = filtered_row[["train_mae"]]
  test_mae = filtered_row[["Mean.absolute.error..MAE."]]
  
  # Create metrics dataframe
  metrics_df = data.frame(
    `Train R2` = train_r2,
    `Test R2` = test_r2,
    `Train RMSE` = train_rmse,
    `Test RMSE` = test_rmse,
    `Train MAE` = train_mae,
    `Test MAE` = test_mae,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  
  # Set column names with proper symbols
  colnames(metrics_df) = c("Train R²", "Test R²", "Train RMSE", "Test RMSE", "Train MAE", "Test MAE")
  
  # Create and return formatted table
  kable(metrics_df, caption = caption, digits = 4, row.names = FALSE, align = "c") %>%
    column_spec(2, border_right = TRUE) %>%   # Line after Test R²
    column_spec(4, border_right = TRUE) %>%   # Line after Test RMSE
    kable_styling(bootstrap_options = c("striped", "hover"))
}

extract_and_display_clustering_metrics = function(model, data, name, peak_obs_lbl, peak_imp_lbl, peak_diff_lbl, caption = "Clustering Statistics") {
  # Function for taking mean, variance, sd, peaks, and observations of observed and imputed models, and giving a dataframe
  
  # Filter data for observed values only
  missingness_0_data = data %>%
    filter(methane_eq_missingness_indicator == 0)
  
  # Create metrics dataframe
  df = data.frame(
    Model = name,
    Observed_Mean = mean(missingness_0_data$methane_eq, na.rm = TRUE),
    Imputed_Mean = mean(data$imputed_methane_eq, na.rm = TRUE),
    Observed_Variance = var(missingness_0_data$methane_eq, na.rm = TRUE),
    Imputed_Variance = var(data$imputed_methane_eq, na.rm = TRUE),
    Observed_SD = sd(missingness_0_data$methane_eq, na.rm = TRUE),
    Imputed_SD = sd(data$imputed_methane_eq, na.rm = TRUE),
    Observed_Peak = peak_obs_lbl,
    Imputed_Peak = peak_imp_lbl,
    Imputed_Minus_Observed_Peak = peak_diff_lbl,
    N_Observed = nrow(missingness_0_data),
    N_Total = nrow(data),
    row.names = NULL,
    stringsAsFactors = FALSE
  )
  
  # Set column names
  colnames(df) = c("Model", "Observed Mean", "Imputed Mean", "Observed Variance", "Imputed Variance", 
                   "Observed SD", "Imputed SD", "Observed Peak", "Imputed Peak", "Difference In Peaks", "Observed # Of Rows", "Total # Of Rows")
  
  # Create and return formatted table
  table_output = knitr::kable(df, caption = caption, digits = 3, row.names = FALSE, align = "c") %>%
    column_spec(3, border_right = TRUE) %>%
    column_spec(5, border_right = TRUE) %>%
    column_spec(7, border_right = TRUE) %>%
    column_spec(10, border_right = TRUE) %>%
    kable_styling(bootstrap_options = c("striped", "hover"))
  
  return(list(data = df, table = table_output))
}