import numpy as np

def generate_matching_data(num_tasks, feature_dim, num_workers, budget_task_bounds=(20, 30), budget_worker_bounds=(50, 60),
                           w_bounds=(0.2, 3), b_bounds=(0, 1),
                           beta_bounds=(0.2, 3), gamma_bounds=(0, 1)):
    
    # Generate task feature weights w and baseline scores b_i
    w = np.random.uniform(w_bounds[0], w_bounds[1], size=feature_dim)
    # b = np.random.uniform(b_bounds[0], b_bounds[1], size=num_tasks)

    # Generate worker features beta_j and baseline abilities gamma_j
    beta = np.random.uniform(beta_bounds[0], beta_bounds[1], size=(num_workers, feature_dim))
    # gamma = np.random.uniform(gamma_bounds[0], gamma_bounds[1], size=num_workers)

    # Generate task feature matrix X (num_tasks, feature_dim)
    X = np.random.uniform(0.2, 3, size=(num_tasks, feature_dim))

    num_30_percent = int(0.3 * num_tasks)
    indices_0_6 = np.random.choice(num_tasks, size=num_30_percent, replace=False)
    indices_0_3 = np.random.choice(
        list(set(range(num_tasks)) - set(indices_0_6)), size=num_30_percent, replace=False
    )

    # 对选中的行进行缩放
    X[indices_0_6] *= 0.6
    X[indices_0_3] *= 1.5

    # Compute the true task scores y_i = log(1 + e^(w^T X_i + b_i))
    true_scores = np.log(1 + np.exp(X @ w))

    # Generate budgets for tasks and workers
    task_budgets = np.random.randint(budget_task_bounds[0], budget_task_bounds[1] + 1, size=num_tasks)
    worker_budgets = np.random.randint(budget_worker_bounds[0], budget_worker_bounds[1] + 1, size=num_workers)

    # Generate worker scores for tasks (num_workers, num_tasks)
    worker_scores = np.full((num_workers, num_tasks), np.nan)
    
    return X, true_scores, task_budgets, worker_budgets, w, beta, worker_scores


def evaluate_test_set(beta, w_2, ground_truth_scores, num_workers, test_indices, task_features):
    test_errors = []
    test_random_errors = []

    for test_task in test_indices:

        sigmas = [np.log(1.5 + np.exp(np.dot(w_2[worker], task_features[test_task]))) for worker in range(num_workers)]
        sigmas_estimate = [np.log(1.5 + np.exp(np.dot(beta[worker], task_features[test_task]))) for worker in range(num_workers)]

        # Get workers who have rated the task
        test_available_workers = [
            worker for worker in range(num_workers)
        ]

        # Select top 3 workers with smallest sigma
        top_workers = sorted(test_available_workers, key=lambda w: sigmas_estimate[w])[:3]
        true_score = ground_truth_scores[test_task]
        predicted_ratings = [np.random.normal(true_score, sigmas[worker]) for worker in top_workers]
        predicted_mean = np.mean(predicted_ratings)  # Avoid NaN issues
        test_errors.append(abs(predicted_mean - true_score))

        # Select 3 random workers for comparison
        random_workers = np.random.choice(test_available_workers, size=3, replace=False)
        random_ratings = [np.random.normal(true_score, sigmas[worker]) for worker in random_workers]
        random_mean = np.mean(random_ratings)
        test_random_errors.append(abs(random_mean - true_score))

    mean_error = np.mean(test_errors) if test_errors else np.nan
    random_error = np.mean(test_random_errors) if test_random_errors else np.nan
    return mean_error, random_error


def train_worker_task_matching(learning_rate, w_1, w_2, ratings_matrix, ground_truth_scores, task_features, task_budgets, worker_budgets, budget, select_size):
    num_workers, num_tasks = ratings_matrix.shape
    feature_dim = task_features.shape[1]
    mean_errors = []
    random_errors = []
    w1_errors = []
    w2_errors = []
    model_errors = []

    worker_weights = np.ones((num_workers, feature_dim))

    w = np.ones(feature_dim)

    all_indices = np.arange(num_tasks)
    train_size = int(0.8 * num_tasks)
    train_indices = np.random.choice(all_indices, size=train_size, replace=False)
    test_indices = np.array([idx for idx in all_indices if idx not in train_indices])

    # Task selection based on uncertainty
    task_uncertainty = np.full(num_tasks, np.inf)

    # Training loop
    while budget > 0:
        available_tasks = [
                task for task in train_indices
                if task_budgets[task] > 0
            ]
        # Compute uncertainty for tasks
        for task in train_indices:
            labeled_workers = [
                worker for worker in range(num_workers)
                if not np.isnan(ratings_matrix[worker, task])
            ]
            if labeled_workers:
                task_uncertainty[task] = np.mean([
                    np.log(1.5 + np.exp(np.dot(worker_weights[worker], task_features[task])))
                    for worker in labeled_workers
            ])

        # Select top Bt tasks with highest uncertainty
        sorted_all_tasks = np.argsort(task_uncertainty)[::-1] 

        selected_tasks = [task for task in sorted_all_tasks if task in available_tasks][:select_size]

        for train_task in selected_tasks:

            # Select workers with highest expertise (smallest sigma)
            worker_sigmas = [
                (worker, np.log(1.5 + np.exp(np.dot(worker_weights[worker], task_features[train_task]))))
                for worker in range(num_workers)
                if np.isnan(ratings_matrix[worker, train_task]) and worker_budgets[worker] > 0
            ]
            worker_sigmas.sort(key=lambda x: x[1])
            selected_workers = [wa[0] for wa in worker_sigmas[:1]]  # Select top 3 workers

            for worker in selected_workers:
                worker_variance = np.log(1.5 + np.exp(np.dot(w_2[worker], task_features[train_task])))
                ratings_matrix[worker, train_task] = np.random.normal(ground_truth_scores[train_task], worker_variance)
                worker_budgets[worker] -= 1

            valid_worker_sigmas = [
            (worker, np.log(1.5 + np.exp(np.dot(worker_weights[worker], task_features[train_task]))))
            for worker in range(num_workers)
            if not np.isnan(ratings_matrix[worker, train_task])
            ]

            # if not valid_worker_sigmas:
            #     estimated_ground_true_score = np.nan
            # else:
            #     # Compute weighted estimate ignoring NaN values
            #     weighted_sum = np.sum([
            #         ratings_matrix[worker, train_task] / (sigma**2 + 1e-8)
            #         for worker, sigma in valid_worker_sigmas
            #     ])
            #     weight_sum = np.sum([1 / (sigma**2 + 1e-8) for _, sigma in valid_worker_sigmas])

            #     estimated_ground_true_score = weighted_sum / weight_sum

            if not valid_worker_sigmas:
                estimated_ground_true_score = np.nan
            else:
                # Compute weighted estimate ignoring NaN values
                weighted_sum = np.sum([
                    ratings_matrix[worker, train_task] / (sigma**2 + 1e-8)
                    for worker, sigma in valid_worker_sigmas
                ])
                weight_sum = np.sum([1 / (sigma**2 + 1e-8) for _, sigma in valid_worker_sigmas])

                estimated_ground_true_score = weighted_sum / weight_sum        

            estimate_true_scores = np.log(1 + np.exp(task_features @ w))

            # Update worker parameters using MLE gradient update
            for worker in selected_workers:
                error = ratings_matrix[worker, train_task] - estimate_true_scores[train_task]
                linear_output = np.dot(worker_weights[worker], task_features[train_task])
                dynamic_sigma = np.log(1.5 + np.exp(linear_output))

                grad_sigma = -(error**2 / (dynamic_sigma**3)) + 1 / (dynamic_sigma)
                grad_activation = np.exp(linear_output) / (1.5 + np.exp(linear_output))
                gradient = grad_sigma * grad_activation * task_features[train_task]
                gradient = np.clip(gradient, -20, 20)
                worker_weights[worker] -= 0.1 * gradient
                

            # Update task model parameters (Softplus function)
            if not np.isnan(estimated_ground_true_score):
                w_error = estimate_true_scores[train_task]- estimated_ground_true_score
                grad_activation = np.exp(np.dot(task_features[train_task], w)) / (1 + np.exp(np.dot(task_features[train_task], w)))
                w -= learning_rate * w_error *  grad_activation * task_features[train_task]
            

            task_budgets[train_task] -= 1

        estimate_true_scores = np.log(1 + np.exp(task_features @ w))

        # Evaluate model
        mean_error, random_error = evaluate_test_set(worker_weights, w_2,ground_truth_scores, num_workers, test_indices, task_features)
        mean_errors.append(mean_error)
        random_errors.append(random_error)
        w1_errors.append(np.mean(abs(w_1 - w)))
        w2_errors.append(np.mean(abs(w_2 - worker_weights)))
        model_errors.append(np.mean(abs(estimate_true_scores - ground_truth_scores)))

        # Decrease total budget
        budget -= select_size

    return w1_errors, w2_errors, random_errors, mean_errors, w, worker_weights, model_errors



def train_worker_task_matching_explore(learning_rate, w_1, w_2, ratings_matrix, ground_truth_scores, task_features, task_budgets, worker_budgets, budget, select_size):
    num_workers, num_tasks = ratings_matrix.shape
    feature_dim = task_features.shape[1]
    mean_errors = []
    random_errors = []
    w1_errors = []
    w2_errors = []
    model_errors = []

    worker_weights = np.ones((num_workers, feature_dim))

    w = np.ones(feature_dim)

    all_indices = np.arange(num_tasks)
    train_size = int(0.8 * num_tasks)
    train_indices = np.random.choice(all_indices, size=train_size, replace=False)
    test_indices = np.array([idx for idx in all_indices if idx not in train_indices])

    # Task selection based on uncertainty
    task_uncertainty = np.full(num_tasks, np.inf)

    ex_budget= budget

    while ex_budget > 0:

        if ex_budget > 0.5 * budget:  # 前50%预算进行随机探索
            available_tasks = [task for task in train_indices if task_budgets[task] > 0]

            selected_tasks = np.random.choice(available_tasks, size=select_size, replace=False)  
        
        else:

            available_tasks = [
                    task for task in train_indices
                    if task_budgets[task] > 0
                ]
            # Compute uncertainty for tasks
            for task in train_indices:
                labeled_workers = [
                    worker for worker in range(num_workers)
                    if not np.isnan(ratings_matrix[worker, task])
                ]
                if labeled_workers:
                    task_uncertainty[task] = np.mean([
                        np.log(1.5 + np.exp(np.dot(worker_weights[worker], task_features[task])))
                        for worker in labeled_workers
                ])

            # Select top Bt tasks with highest uncertainty
            sorted_all_tasks = np.argsort(task_uncertainty)[::-1] 

            selected_tasks = [task for task in sorted_all_tasks if task in available_tasks][:select_size]

        for train_task in selected_tasks:

            if ex_budget > 0.5 * budget: 
                available_workers = [worker for worker in range(num_workers) if np.isnan(ratings_matrix[worker, train_task]) and worker_budgets[worker] > 0]
                selected_workers = np.random.choice(available_workers, size=1, replace=False)  
            
            else:

            # Select workers with highest expertise (smallest sigma)
                worker_sigmas = [
                    (worker, np.log(1.5 + np.exp(np.dot(worker_weights[worker], task_features[train_task]))))
                    for worker in range(num_workers)
                    if np.isnan(ratings_matrix[worker, train_task]) and worker_budgets[worker] > 0
                ]
                worker_sigmas.sort(key=lambda x: x[1])
                selected_workers = [wa[0] for wa in worker_sigmas[:1]]  # Select top 3 workers

            for worker in selected_workers:
                worker_variance = np.log(1.5 + np.exp(np.dot(w_2[worker], task_features[train_task])))
                ratings_matrix[worker, train_task] = np.random.normal(ground_truth_scores[train_task], worker_variance)
                worker_budgets[worker] -= 1

            valid_worker_sigmas = [
            (worker, np.log(1.5 + np.exp(np.dot(worker_weights[worker], task_features[train_task]))))
            for worker in range(num_workers)
            if not np.isnan(ratings_matrix[worker, train_task])
            ]

            if not valid_worker_sigmas:
                estimated_ground_true_score = np.nan
            else:
                # Compute weighted estimate ignoring NaN values
                weighted_sum = np.sum([
                    ratings_matrix[worker, train_task] / (sigma**2 + 1e-8)
                    for worker, sigma in valid_worker_sigmas
                ])
                weight_sum = np.sum([1 / (sigma**2 + 1e-8) for _, sigma in valid_worker_sigmas])

                estimated_ground_true_score = weighted_sum / weight_sum    

            estimate_true_scores = np.log(1 + np.exp(task_features @ w))
            # Update worker parameters using MLE gradient update
            for worker in selected_workers:
                error = ratings_matrix[worker, train_task] - estimate_true_scores[train_task]
                linear_output = np.dot(worker_weights[worker], task_features[train_task])
                dynamic_sigma = np.log(1.5 + np.exp(linear_output))

                grad_sigma = -(error**2 / (dynamic_sigma**3)) + 1 / (dynamic_sigma)
                grad_activation = np.exp(linear_output) / (1.5 + np.exp(linear_output))
                gradient = grad_sigma * grad_activation * task_features[train_task]
                gradient = np.clip(gradient, -20, 20)
                worker_weights[worker] -= learning_rate * gradient
                

            # Update task model parameters (Softplus function)
            if not np.isnan(estimated_ground_true_score):
                predicted_score = np.log(1 + np.exp(np.dot(task_features[train_task], w)))
                w_error = predicted_score - estimated_ground_true_score
                grad_activation = np.exp(np.dot(task_features[train_task], w)) / (1 + np.exp(np.dot(task_features[train_task], w)))
                w -= 0.01 * w_error *  grad_activation * task_features[train_task]
            

            task_budgets[train_task] -= 1

        estimate_true_scores = np.log(1 + np.exp(task_features @ w))

        # Evaluate model
        mean_error, random_error = evaluate_test_set(worker_weights, w_2,ground_truth_scores, num_workers, test_indices, task_features)
        mean_errors.append(mean_error)
        random_errors.append(random_error)
        w1_errors.append(np.mean(abs(w_1 - w)))
        w2_errors.append(np.mean(abs(w_2 - worker_weights)))
        model_errors.append(np.mean(abs(estimate_true_scores - ground_truth_scores)))

        # Decrease total budget
        ex_budget -= select_size


    return w1_errors, w2_errors, random_errors, mean_errors, w, worker_weights, model_errors



def train_worker_task_matching_explore_worker(learning_rate, w_1, w_2, ratings_matrix, ground_truth_scores, task_features, task_budgets, worker_budgets, budget, select_size):
    num_workers, num_tasks = ratings_matrix.shape
    feature_dim = task_features.shape[1]
    mean_errors = []
    random_errors = []
    w1_errors = []
    w2_errors = []
    model_errors = []

    worker_weights = np.ones((num_workers, feature_dim))

    w = np.ones(feature_dim)

    all_indices = np.arange(num_tasks)
    train_size = int(0.8 * num_tasks)
    train_indices = np.random.choice(all_indices, size=train_size, replace=False)
    test_indices = np.array([idx for idx in all_indices if idx not in train_indices])

    # Task selection based on uncertainty
    task_uncertainty = np.full(num_tasks, np.inf)

    ex_budget= budget

    while ex_budget > 0:

        available_tasks = [
                task for task in train_indices
                if task_budgets[task] > 0
            ]
        # Compute uncertainty for tasks
        for task in train_indices:
            labeled_workers = [
                worker for worker in range(num_workers)
                if not np.isnan(ratings_matrix[worker, task])
            ]
            if labeled_workers:
                task_uncertainty[task] = np.mean([
                    np.log(1.5 + np.exp(np.dot(worker_weights[worker], task_features[task])))
                    for worker in labeled_workers
            ])

        # Select top Bt tasks with highest uncertainty
        sorted_all_tasks = np.argsort(task_uncertainty)[::-1] 

        selected_tasks = [task for task in sorted_all_tasks if task in available_tasks][:select_size]

        for train_task in selected_tasks:

            if ex_budget > 0.5 * budget: 
                available_workers = [worker for worker in range(num_workers) if np.isnan(ratings_matrix[worker, train_task]) and worker_budgets[worker] > 0]
                selected_workers = np.random.choice(available_workers, size=1, replace=False)  
            
            else:

            # Select workers with highest expertise (smallest sigma)
                worker_sigmas = [
                    (worker, np.log(1.5 + np.exp(np.dot(worker_weights[worker], task_features[train_task]))))
                    for worker in range(num_workers)
                    if np.isnan(ratings_matrix[worker, train_task]) and worker_budgets[worker] > 0
                ]
                worker_sigmas.sort(key=lambda x: x[1])
                selected_workers = [wa[0] for wa in worker_sigmas[:1]]  # Select top 3 workers

            for worker in selected_workers:
                worker_variance = np.log(1.5 + np.exp(np.dot(w_2[worker], task_features[train_task])))
                ratings_matrix[worker, train_task] = np.random.normal(ground_truth_scores[train_task], worker_variance)
                worker_budgets[worker] -= 1

            valid_worker_sigmas = [
            (worker, np.log(1.5 + np.exp(np.dot(worker_weights[worker], task_features[train_task]))))
            for worker in range(num_workers)
            if not np.isnan(ratings_matrix[worker, train_task])
            ]

            if not valid_worker_sigmas:
                estimated_ground_true_score = np.nan
            else:
                # Compute weighted estimate ignoring NaN values
                weighted_sum = np.sum([
                    ratings_matrix[worker, train_task] / (sigma**2 + 1e-8)
                    for worker, sigma in valid_worker_sigmas
                ])
                weight_sum = np.sum([1 / (sigma**2 + 1e-8) for _, sigma in valid_worker_sigmas])

                estimated_ground_true_score = weighted_sum / weight_sum    

            estimate_true_scores = np.log(1 + np.exp(task_features @ w))
            # Update worker parameters using MLE gradient update
            for worker in selected_workers:
                error = ratings_matrix[worker, train_task] - estimate_true_scores[train_task]
                linear_output = np.dot(worker_weights[worker], task_features[train_task])
                dynamic_sigma = np.log(1.5 + np.exp(linear_output))

                grad_sigma = -(error**2 / (dynamic_sigma**3)) + 1 / (dynamic_sigma)
                grad_activation = np.exp(linear_output) / (1.5 + np.exp(linear_output))
                gradient = grad_sigma * grad_activation * task_features[train_task]
                gradient = np.clip(gradient, -20, 20)
                worker_weights[worker] -= learning_rate * gradient
                

            # Update task model parameters (Softplus function)
            if not np.isnan(estimated_ground_true_score):
                predicted_score = np.log(1 + np.exp(np.dot(task_features[train_task], w)))
                w_error = predicted_score - estimated_ground_true_score
                grad_activation = np.exp(np.dot(task_features[train_task], w)) / (1 + np.exp(np.dot(task_features[train_task], w)))
                w -= 0.01 * w_error *  grad_activation * task_features[train_task]
            

            task_budgets[train_task] -= 1

        estimate_true_scores = np.log(1 + np.exp(task_features @ w))

        # Evaluate model
        mean_error, random_error = evaluate_test_set(worker_weights, w_2,ground_truth_scores, num_workers, test_indices, task_features)
        mean_errors.append(mean_error)
        random_errors.append(random_error)
        w1_errors.append(np.mean(abs(w_1 - w)))
        w2_errors.append(np.mean(abs(w_2 - worker_weights)))
        model_errors.append(np.mean(abs(estimate_true_scores - ground_truth_scores)))

        # Decrease total budget
        ex_budget -= select_size


    return w1_errors, w2_errors, random_errors, mean_errors, w, worker_weights, model_errors



def train_worker_task_matching_explore_all(learning_rate, w_1, w_2, ratings_matrix, ground_truth_scores, task_features, task_budgets, worker_budgets, budget, select_size):
    num_workers, num_tasks = ratings_matrix.shape
    feature_dim = task_features.shape[1]
    mean_errors = []
    random_errors = []
    w1_errors = []
    w2_errors = []
    model_errors = []

    worker_weights = np.ones((num_workers, feature_dim))

    w = np.ones(feature_dim)

    all_indices = np.arange(num_tasks)
    train_size = int(0.8 * num_tasks)
    train_indices = np.random.choice(all_indices, size=train_size, replace=False)
    test_indices = np.array([idx for idx in all_indices if idx not in train_indices])

    # Task selection based on uncertainty
    task_uncertainty = np.full(num_tasks, np.inf)

    ex_budget= budget

    while ex_budget > 0:

        available_tasks = [task for task in train_indices if task_budgets[task] > 0]

        selected_tasks = np.random.choice(available_tasks, size=select_size, replace=False)  

        for train_task in selected_tasks:

            available_workers = [worker for worker in range(num_workers) if np.isnan(ratings_matrix[worker, train_task]) and worker_budgets[worker] > 0]
            selected_workers = np.random.choice(available_workers, size=1, replace=False)  

            for worker in selected_workers:
                worker_variance = np.log(1.5 + np.exp(np.dot(w_2[worker], task_features[train_task])))
                ratings_matrix[worker, train_task] = np.random.normal(ground_truth_scores[train_task], worker_variance)
                worker_budgets[worker] -= 1

            valid_worker_sigmas = [
            (worker, np.log(1.5 + np.exp(np.dot(worker_weights[worker], task_features[train_task]))))
            for worker in range(num_workers)
            if not np.isnan(ratings_matrix[worker, train_task])
            ]

            if not valid_worker_sigmas:
                estimated_ground_true_score = np.nan
            else:
                # Compute weighted estimate ignoring NaN values
                weighted_sum = np.sum([
                    ratings_matrix[worker, train_task] / (sigma**2 + 1e-8)
                    for worker, sigma in valid_worker_sigmas
                ])
                weight_sum = np.sum([1 / (sigma**2 + 1e-8) for _, sigma in valid_worker_sigmas])

                estimated_ground_true_score = weighted_sum / weight_sum    

            estimate_true_scores = np.log(1 + np.exp(task_features @ w))
            # Update worker parameters using MLE gradient update
            for worker in selected_workers:
                error = ratings_matrix[worker, train_task] - estimate_true_scores[train_task]
                linear_output = np.dot(worker_weights[worker], task_features[train_task])
                dynamic_sigma = np.log(1.5 + np.exp(linear_output))

                grad_sigma = -(error**2 / (dynamic_sigma**3)) + 1 / (dynamic_sigma)
                grad_activation = np.exp(linear_output) / (1.5 + np.exp(linear_output))
                gradient = grad_sigma * grad_activation * task_features[train_task]
                gradient = np.clip(gradient, -20, 20)
                worker_weights[worker] -= learning_rate * gradient
                

            # Update task model parameters (Softplus function)
            if not np.isnan(estimated_ground_true_score):
                predicted_score = np.log(1 + np.exp(np.dot(task_features[train_task], w)))
                w_error = predicted_score - estimated_ground_true_score
                grad_activation = np.exp(np.dot(task_features[train_task], w)) / (1 + np.exp(np.dot(task_features[train_task], w)))
                w -= 0.005 * w_error *  grad_activation * task_features[train_task]
            

            task_budgets[train_task] -= 1

        estimate_true_scores = np.log(1 + np.exp(task_features @ w))

        # Evaluate model
        mean_error, random_error = evaluate_test_set(worker_weights, w_2,ground_truth_scores, num_workers, test_indices, task_features)
        mean_errors.append(mean_error)
        random_errors.append(random_error)
        w1_errors.append(np.mean(abs(w_1 - w)))
        w2_errors.append(np.mean(abs(w_2 - worker_weights)))
        model_errors.append(np.mean(abs(estimate_true_scores - ground_truth_scores)))

        # Decrease total budget
        ex_budget -= select_size


    return w1_errors, w2_errors, random_errors, mean_errors, w, worker_weights, model_errors



def train_worker_task_matching_true_score(learning_rate, w_1, w_2, ratings_matrix, ground_truth_scores, task_features, task_budgets, worker_budgets, budget, select_size):
    num_workers, num_tasks = ratings_matrix.shape
    feature_dim = task_features.shape[1]
    mean_errors = []
    random_errors = []
    w1_errors = []
    w2_errors = []
    model_errors = []

    worker_weights = np.ones((num_workers, feature_dim))

    w = np.ones(feature_dim)

    all_indices = np.arange(num_tasks)
    train_size = int(0.8 * num_tasks)
    train_indices = np.random.choice(all_indices, size=train_size, replace=False)
    test_indices = np.array([idx for idx in all_indices if idx not in train_indices])

    # Task selection based on uncertainty
    task_uncertainty = np.full(num_tasks, np.inf)

    # Training loop
    while budget > 0:
        available_tasks = [
                task for task in train_indices
                if task_budgets[task] > 0
            ]
        # Compute uncertainty for tasks
        for task in train_indices:
            labeled_workers = [
                worker for worker in range(num_workers)
                if not np.isnan(ratings_matrix[worker, task])
            ]
            if labeled_workers:
                task_uncertainty[task] = np.mean([
                    np.log(1.5 + np.exp(np.dot(worker_weights[worker], task_features[task])))
                    for worker in labeled_workers
            ])

        # Select top Bt tasks with highest uncertainty
        sorted_all_tasks = np.argsort(task_uncertainty)[::-1] 

        selected_tasks = [task for task in sorted_all_tasks if task in available_tasks][:select_size]

        for train_task in selected_tasks:

            # Select workers with highest expertise (smallest sigma)
            worker_sigmas = [
                (worker, np.log(1.5 + np.exp(np.dot(worker_weights[worker], task_features[train_task]))))
                for worker in range(num_workers)
                if np.isnan(ratings_matrix[worker, train_task]) and worker_budgets[worker] > 0
            ]
            worker_sigmas.sort(key=lambda x: x[1])
            selected_workers = [wa[0] for wa in worker_sigmas[:1]]  # Select top 3 workers

            for worker in selected_workers:
                worker_variance = np.log(1.5 + np.exp(np.dot(w_2[worker], task_features[train_task])))
                ratings_matrix[worker, train_task] = np.random.normal(ground_truth_scores[train_task], worker_variance)
                worker_budgets[worker] -= 1

            valid_worker_sigmas = [
            (worker, np.log(1.5 + np.exp(np.dot(worker_weights[worker], task_features[train_task]))))
            for worker in range(num_workers)
            if not np.isnan(ratings_matrix[worker, train_task])
            ]

            # if not valid_worker_sigmas:
            #     estimated_ground_true_score = np.nan
            # else:
            #     # Compute weighted estimate ignoring NaN values
            #     weighted_sum = np.sum([
            #         ratings_matrix[worker, train_task] / (sigma**2 + 1e-8)
            #         for worker, sigma in valid_worker_sigmas
            #     ])
            #     weight_sum = np.sum([1 / (sigma**2 + 1e-8) for _, sigma in valid_worker_sigmas])

            #     estimated_ground_true_score = weighted_sum / weight_sum

            if not valid_worker_sigmas:
                estimated_ground_true_score = np.nan
            else:
                # Compute weighted estimate ignoring NaN values
                weighted_sum = np.sum([
                    ratings_matrix[worker, train_task] / (sigma**2 + 1e-8)
                    for worker, sigma in valid_worker_sigmas
                ])
                weight_sum = np.sum([1 / (sigma**2 + 1e-8) for _, sigma in valid_worker_sigmas])

                estimated_ground_true_score = weighted_sum / weight_sum        

            estimate_true_scores = np.log(1 + np.exp(task_features @ w))

            # Update worker parameters using MLE gradient update
            for worker in selected_workers:
                error = ratings_matrix[worker, train_task] - estimate_true_scores[train_task]
                linear_output = np.dot(worker_weights[worker], task_features[train_task])
                dynamic_sigma = np.log(1.5 + np.exp(linear_output))

                grad_sigma = -(error**2 / (dynamic_sigma**3)) + 1 / (dynamic_sigma)
                grad_activation = np.exp(linear_output) / (1.5 + np.exp(linear_output))
                gradient = grad_sigma * grad_activation * task_features[train_task]
                gradient = np.clip(gradient, -20, 20)
                worker_weights[worker] -= learning_rate * gradient
                

            # Update task model parameters (Softplus function)
            if not np.isnan(estimated_ground_true_score):
                w_error = estimate_true_scores[train_task]- ground_truth_scores[train_task]
                grad_activation = np.exp(np.dot(task_features[train_task], w)) / (1 + np.exp(np.dot(task_features[train_task], w)))
                w -= 0.005 * w_error *  grad_activation * task_features[train_task]
            

            task_budgets[train_task] -= 1

        estimate_true_scores = np.log(1 + np.exp(task_features @ w))

        # Evaluate model
        mean_error, random_error = evaluate_test_set(worker_weights, w_2,ground_truth_scores, num_workers, test_indices, task_features)
        mean_errors.append(mean_error)
        random_errors.append(random_error)
        w1_errors.append(np.mean(abs(w_1 - w)))
        w2_errors.append(np.mean(abs(w_2 - worker_weights)))
        model_errors.append(np.mean(abs(estimate_true_scores - ground_truth_scores)))

        # Decrease total budget
        budget -= select_size

    return w1_errors, w2_errors, random_errors, mean_errors, w, worker_weights, model_errors





def train_worker_task_matching_explore_true_score(w_1, w_2, ratings_matrix, ground_truth_scores, task_features, task_budgets, worker_budgets, budget, select_size):
    num_workers, num_tasks = ratings_matrix.shape
    feature_dim = task_features.shape[1]
    mean_errors = []
    random_errors = []
    w1_errors = []
    w2_errors = []
    model_errors = []

    worker_weights = np.ones((num_workers, feature_dim))

    w = np.ones(feature_dim)

    all_indices = np.arange(num_tasks)
    train_size = int(0.8 * num_tasks)
    train_indices = np.random.choice(all_indices, size=train_size, replace=False)
    test_indices = np.array([idx for idx in all_indices if idx not in train_indices])

    # Task selection based on uncertainty
    task_uncertainty = np.full(num_tasks, np.inf)

    ex_budget= budget

    while ex_budget > 0:

        if ex_budget > 0.5 * budget:  # 前50%预算进行随机探索
            available_tasks = [task for task in train_indices if task_budgets[task] > 0]

            selected_tasks = np.random.choice(available_tasks, size=select_size, replace=False)  
        
        else:

            available_tasks = [
                    task for task in train_indices
                    if task_budgets[task] > 0
                ]
            # Compute uncertainty for tasks
            for task in train_indices:
                labeled_workers = [
                    worker for worker in range(num_workers)
                    if not np.isnan(ratings_matrix[worker, task])
                ]
                if labeled_workers:
                    task_uncertainty[task] = np.mean([
                        np.log(1.5 + np.exp(np.dot(worker_weights[worker], task_features[task])))
                        for worker in labeled_workers
                ])

            # Select top Bt tasks with highest uncertainty
            sorted_all_tasks = np.argsort(task_uncertainty)[::-1] 

            selected_tasks = [task for task in sorted_all_tasks if task in available_tasks][:select_size]

        for train_task in selected_tasks:

            if ex_budget > 0.5 * budget: 
                available_workers = [worker for worker in range(num_workers) if np.isnan(ratings_matrix[worker, train_task]) and worker_budgets[worker] > 0]
                selected_workers = np.random.choice(available_workers, size=1, replace=False)  
            
            else:

            # Select workers with highest expertise (smallest sigma)
                worker_sigmas = [
                    (worker, np.log(1.5 + np.exp(np.dot(worker_weights[worker], task_features[train_task]))))
                    for worker in range(num_workers)
                    if np.isnan(ratings_matrix[worker, train_task]) and worker_budgets[worker] > 0
                ]
                worker_sigmas.sort(key=lambda x: x[1])
                selected_workers = [wa[0] for wa in worker_sigmas[:1]]  # Select top 3 workers

            for worker in selected_workers:
                worker_variance = np.log(1.5 + np.exp(np.dot(w_2[worker], task_features[train_task])))
                ratings_matrix[worker, train_task] = np.random.normal(ground_truth_scores[train_task], worker_variance)
                worker_budgets[worker] -= 1

            valid_worker_sigmas = [
            (worker, np.log(1.5 + np.exp(np.dot(worker_weights[worker], task_features[train_task]))))
            for worker in range(num_workers)
            if not np.isnan(ratings_matrix[worker, train_task])
            ]

            if not valid_worker_sigmas:
                estimated_ground_true_score = np.nan
            else:
                # Compute weighted estimate ignoring NaN values
                weighted_sum = np.sum([
                    ratings_matrix[worker, train_task] / (sigma**2 + 1e-8)
                    for worker, sigma in valid_worker_sigmas
                ])
                weight_sum = np.sum([1 / (sigma**2 + 1e-8) for _, sigma in valid_worker_sigmas])

                estimated_ground_true_score = weighted_sum / weight_sum    

            estimate_true_scores = np.log(1 + np.exp(task_features @ w))
            # Update worker parameters using MLE gradient update
            for worker in selected_workers:
                error = ratings_matrix[worker, train_task] - estimate_true_scores[train_task]
                linear_output = np.dot(worker_weights[worker], task_features[train_task])
                dynamic_sigma = np.log(1.5 + np.exp(linear_output))

                grad_sigma = -(error**2 / (dynamic_sigma**3)) + 1 / (dynamic_sigma)
                grad_activation = np.exp(linear_output) / (1.5 + np.exp(linear_output))
                gradient = grad_sigma * grad_activation * task_features[train_task]
                gradient = np.clip(gradient, -20, 20)
                worker_weights[worker] -= 0.05 * gradient
                

            # Update task model parameters (Softplus function)
            if not np.isnan(estimated_ground_true_score):
                predicted_score = np.log(1 + np.exp(np.dot(task_features[train_task], w)))
                w_error = predicted_score - ground_truth_scores[train_task]
                grad_activation = np.exp(np.dot(task_features[train_task], w)) / (1 + np.exp(np.dot(task_features[train_task], w)))
                w -= 0.01 * w_error *  grad_activation * task_features[train_task]
            

            task_budgets[train_task] -= 1

        estimate_true_scores = np.log(1 + np.exp(task_features @ w))

        # Evaluate model
        mean_error, random_error = evaluate_test_set(worker_weights, w_2,ground_truth_scores, num_workers, test_indices, task_features)
        mean_errors.append(mean_error)
        random_errors.append(random_error)
        w1_errors.append(np.mean(abs(w_1 - w)))
        w2_errors.append(np.mean(abs(w_2 - worker_weights)))
        model_errors.append(np.mean(abs(estimate_true_scores - ground_truth_scores)))

        # Decrease total budget
        ex_budget -= select_size


    return w1_errors, w2_errors, random_errors, mean_errors, w, worker_weights, model_errors

