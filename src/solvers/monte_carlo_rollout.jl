const PROB_THRESHOLD = 1e-10

function rollout_policy(particles, weights, K, n, A, p_a, eta, alpha, 
                          lookahead_horizon, rollout_horizon, num_simulations, N_samples, threshold)
    """
    Implements Monte Carlo rollout algorithm using component-wise observation sampling
    
    Args:
    - particles: Matrix where each column is a particle (state vector)
    - weights: Particle weights
    - K: Number of components
    - n: Maximum observation value per component
    - A: Transition matrix
    - p_a: Transition probability
    - C: Cost matrix
    - alpha: Discount factor
    - lookahead_horizon: Lookahead horizon
    - rollout_horizon: Rollout horizon
    - num_simulations: Number of Monte Carlo simulations
    - N: Number of sample observations for lookahead
    - threshold: Threshold for base policy decision making
    
    Returns:
    - Rollout control vector
    """
    
    intrusion_dist, no_intrusion_dist = RecoveryPOMDP.generate_component_observation_distributions(n)
    
    best_control_vec = Tuple(zeros(Int, K))
    best_value = Inf
    
    # Generate all possible control vectors
    total_controls = 2^K
    control_count = 0
    
    for u_vec in Iterators.product(fill(0:1, K)...)
        control_count += 1
        remaining = total_controls - control_count + 1
        println("Evaluating control vector $control_count/$total_controls ($(remaining-1) remaining): $u_vec")
        
        u_tuple = Tuple(u_vec)
        q_value = compute_q_value(particles, weights, u_tuple, K,
                                    A, p_a, eta, alpha, lookahead_horizon, rollout_horizon, 
                                    num_simulations, N_samples, threshold, intrusion_dist, no_intrusion_dist)
        
        if q_value < best_value
            best_value = q_value
            best_control_vec = u_tuple
        end
    end
    
    return best_control_vec
end

function compute_q_value(particles, weights, u_vec, K, A, p_a, eta, alpha, 
                           lookahead_horizon, rollout_horizon, num_simulations, N_samples, threshold,
                           intrusion_dist, no_intrusion_dist)
    """
    Computes Q-value using Monte Carlo sampling with component-wise observation sampling
    """
    
    immediate_cost = compute_particle_expected_cost(particles, weights, u_vec, eta, K)
    
    if lookahead_horizon == 0
        if rollout_horizon > 0
            rollout_cost = 0.0
            for sim in 1:num_simulations
                # Sample observation after applying control u_vec
                o_sample = BootstrapParticleFilter.sample_observation_from_particles(particles, weights, u_vec, K, A, p_a, intrusion_dist, no_intrusion_dist)
                # Update particles and weights after applying control u_vec and observing o_sample
                new_particles, new_weights = BootstrapParticleFilter.particle_filter_update(particles, weights, u_vec, o_sample, K, A, p_a, intrusion_dist, no_intrusion_dist)
                # Run rollout simulation from the updated particle set
                sim_cost = simulate_rollout(new_particles, new_weights, eta, K, A, p_a, alpha, rollout_horizon, threshold, intrusion_dist, no_intrusion_dist)
                rollout_cost += sim_cost
            end
            rollout_cost /= num_simulations
            return immediate_cost + alpha * rollout_cost + 
                   alpha^(rollout_horizon + 1) * terminal_cost(particles, weights, eta, K, threshold)
        else
            return immediate_cost + alpha * terminal_cost(particles, weights, eta, K, threshold)
        end
    end
    
    future_cost = 0.0
    
    sampled_observations = BootstrapParticleFilter.sample_observations_from_particles(particles, weights, u_vec, K, N_samples, A, p_a, intrusion_dist, no_intrusion_dist)
    
    @inbounds for o_vec in sampled_observations
        new_particles, new_weights = BootstrapParticleFilter.particle_filter_update(particles, weights, u_vec, o_vec, K, A, p_a, intrusion_dist, no_intrusion_dist)
        
        if sum(new_weights) > PROB_THRESHOLD
            min_expected_cost = Inf
            # Generate all possible next control vectors
            for u_next_vec in Iterators.product(fill(0:1, K)...)
                u_next_tuple = Tuple(u_next_vec)
                q_next = compute_q_value(new_particles, new_weights, u_next_tuple, K,
                                          A, p_a, eta, alpha, lookahead_horizon - 1, rollout_horizon, 
                                          num_simulations, N_samples, threshold, intrusion_dist, no_intrusion_dist)
                min_expected_cost = min(min_expected_cost, q_next)
            end
            
            obs_prob = compute_observation_probability(o_vec, new_particles, new_weights, K, intrusion_dist, no_intrusion_dist)
            future_cost += obs_prob * alpha * min_expected_cost
        end
    end
    
    return immediate_cost + future_cost
end

function simulate_rollout(particles, weights, eta, K, A, p_a, alpha, horizon, threshold,
                            intrusion_dist, no_intrusion_dist)
    """
    Monte Carlo rollout simulation using component-wise observation sampling
    """
    
    particles_copy = copy(particles)
    weights_copy = copy(weights)
    total_cost = 0.0
    alpha_power = 1.0
    
    @inbounds for t in 1:horizon
        u_t_vec = base_policy(particles_copy, weights_copy, K, threshold)
        cost_t = compute_particle_expected_cost(particles_copy, weights_copy, u_t_vec, eta, K)
        total_cost += alpha_power * cost_t
        alpha_power *= alpha
        
        if t < horizon
            o_t = BootstrapParticleFilter.sample_observation_from_particles(particles_copy, weights_copy, u_t_vec, K, A, p_a, intrusion_dist, no_intrusion_dist)
            particles_copy, weights_copy = BootstrapParticleFilter.particle_filter_update(particles_copy, weights_copy, 
                                                                       u_t_vec, o_t, K, A, p_a, intrusion_dist, no_intrusion_dist)
        end
    end
    
    return total_cost
end

function terminal_cost(particles, weights, eta, K, threshold)
    """
    Compute terminal cost using particle representation
    """
    return 0
    #u_terminal_vec = base_policy(particles, weights, K, threshold)
    #return compute_particle_expected_cost(particles, weights, u_terminal_vec, eta, K)
end

function base_policy(particles, weights, K, threshold)
    """
    Threshold-based base policy that sets control u=1 for each component 
    if its belief of being compromised is greater than the given threshold
    """
    M = size(particles, 2)
    
    # Compute belief probabilities for each component being compromised
    component_beliefs = zeros(K)
    
    @inbounds for i in 1:M
        state_vec = Tuple(particles[:, i])
        weight = weights[i]
        
        # Add weight to compromised belief for each component that is compromised (state=1)
        for k in 1:K
            if state_vec[k] == 1
                component_beliefs[k] += weight
            end
        end
    end
    
    # Create control vector based on threshold policy
    control_vec = Vector{Int}(undef, K)
    @inbounds for k in 1:K
        control_vec[k] = component_beliefs[k] > threshold ? 1 : 0
    end
    
    # Convert control vector to control tuple
    return Tuple(control_vec)
end

function compute_observation_probability(o_vec, particles, weights, K,
                                        intrusion_dist, no_intrusion_dist)
    """
    Compute the probability of an observation vector using the factorized structure
    """
    M = size(particles, 2)
    total_prob = 0.0
    
    @inbounds for i in 1:M
        state_vec = Tuple(particles[:, i])
        
        obs_likelihood = RecoveryPOMDP.compute_observation_likelihood(o_vec, state_vec, intrusion_dist, no_intrusion_dist)
        total_prob += weights[i] * obs_likelihood
    end
    
    return total_prob
end

function compute_particle_expected_cost(particles, weights, u_vec, eta, K)
    """
    Compute expected cost using particle representation with direct cost computation
    """
    
    M = size(particles, 2)
    expected_cost = 0.0
    
    @inbounds for i in 1:M
        state_vec = Tuple(particles[:, i])
        cost = RecoveryPOMDP.compute_cost(state_vec, u_vec, eta)
        expected_cost += weights[i] * cost
    end
    
    return expected_cost
end

function run_rollout_simulation(initial_state_vec, K, n, A, p_a, eta,
                                  alpha, lookahead_horizon, rollout_horizon,
                                  num_simulations, N_samples, M, T, eval_samples, threshold)
    """
    Monte Carlo rollout simulation for T time steps using component-wise observation sampling
    
    Args:
    - initial_state_vec: Initial state vector (deterministic)
    - K: Number of components
    - n: Maximum observation value per component
    - A: Transition matrix
    - p_a: Transition probability
    - C: Cost matrix
    - alpha: Discount factor
    - lookahead_horizon: Lookahead horizon
    - rollout_horizon: Rollout horizon
    - num_simulations: Number of Monte Carlo simulations
    - N: Number of sample observations for lookahead
    - M: Number of particles
    - T: Number of time steps to simulate
    - eval_samples: Number of evaluation samples to run
    - threshold: Threshold for base policy decision making
    
    Returns:
    - Average total cost across all evaluation samples
    """
    
    intrusion_dist, no_intrusion_dist = RecoveryPOMDP.generate_component_observation_distributions(n)
    total_costs = Vector{Float64}(undef, eval_samples)
    
    @inbounds for sample in 1:eval_samples
        # Initialize particles with the deterministic initial state
        particles = Matrix{Int}(undef, K, M)
        for i in 1:M
            particles[:, i] = collect(initial_state_vec)
        end
        weights = fill(1.0 / M, M)
        
        true_state_vec = initial_state_vec
        
        total_cost = 0.0
        alpha_power = 1.0
        
        @inbounds for t in 1:T
            u_t_vec = rollout_policy(particles, weights, K, n, A, p_a, eta,
                                   alpha, lookahead_horizon, rollout_horizon, num_simulations, N_samples, threshold)
            
            cost_t = RecoveryPOMDP.compute_cost(true_state_vec, u_t_vec, eta)
            total_cost += alpha_power * cost_t
            alpha_power *= alpha
            
            if t < T
                o_t = RecoveryPOMDP.sample_observation_vector(true_state_vec, intrusion_dist, no_intrusion_dist)
                
                true_state_vec = RecoveryPOMDP.sample_next_state_vector(true_state_vec, u_t_vec, p_a, A)
                
                particles, weights = BootstrapParticleFilter.particle_filter_update(particles, weights, 
                                                                           u_t_vec, o_t, K, A, p_a, intrusion_dist, no_intrusion_dist)
            end
        end
        
        total_costs[sample] = total_cost
                
        if sample % 1 == 0 || sample == eval_samples
            current_avg = sum(total_costs[1:sample]) / sample
            println("Sample $sample/$eval_samples, Current average cost: $(round(current_avg, digits=4))")
        end
    end

    return sum(total_costs) / eval_samples
end
