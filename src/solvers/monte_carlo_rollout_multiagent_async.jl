const PROB_THRESHOLD = 1e-10

function rollout_policy(particles, weights, K, n, A, p_a, alpha, 
                          lookahead_horizon, rollout_horizon, num_simulations, N, threshold, eta)
    """
    Implements Parallel Multiagent Monte Carlo rollout algorithm using component-wise optimization
    
    Each component of the control vector is optimized in parallel:
    1. All components k=1..K are optimized simultaneously 
    2. Each component assumes all other components will use the base policy
    3. Optimize only component k's control value independently
    
    This parallel approach trades off some coordination between components for faster execution,
    especially beneficial when K is large and multiple CPU cores are available.
    
    Args:
    - particles: Matrix where each column is a particle (state vector)
    - weights: Particle weights
    - K: Number of components
    - n: Maximum observation value per component
    - A: Transition matrix
    - p_a: Transition probability
    - alpha: Discount factor
    - lookahead_horizon: Lookahead horizon
    - rollout_horizon: Rollout horizon
    - num_simulations: Number of Monte Carlo simulations
    - N: Number of sample observations for lookahead
    - threshold: Threshold for base policy
    - eta: Cost parameter
    
    Returns:
    - Rollout control vector
    """
    
    intrusion_dist, no_intrusion_dist = RecoveryPOMDP.generate_component_observation_distributions(n)
    
    base_control_vec = get_base_policy_vector(particles, weights, K, threshold)
    
    optimal_control_vec = Vector{Int}(undef, K)
    
    Threads.@threads for k in 1:K
        best_component_value = base_control_vec[k]
        best_q_value = Inf
        
        @inbounds for u_k in [0, 1]
            test_control_vec = copy(base_control_vec)
            test_control_vec[k] = u_k
            
            u_vec = NTuple{K,Int}(test_control_vec)
            
            q_value = compute_q_value(particles, weights, u_vec, k, base_control_vec, K,
                                                A, p_a, alpha, lookahead_horizon, rollout_horizon, 
                                                num_simulations, N, threshold, intrusion_dist, no_intrusion_dist, eta)
            
            if q_value < best_q_value
                best_q_value = q_value
                best_component_value = u_k
            end
        end
        
        optimal_control_vec[k] = best_component_value
    end
    
    return NTuple{K,Int}(optimal_control_vec)
end

function compute_q_value(particles, weights, u_vec, 
                                   component_k, partial_control_vec,
                                   K, A, p_a, alpha, 
                                   lookahead_horizon, rollout_horizon, num_simulations, N, threshold,
                                   intrusion_dist, no_intrusion_dist, eta)
    """
    Computes Q-value for multiagent rollout where we're optimizing component k
    and assuming remaining components follow base policy
    """
    
    immediate_cost = compute_particle_expected_cost(particles, weights, u_vec, eta, K)
    
    if lookahead_horizon == 0
        if rollout_horizon > 0
            rollout_cost = 0.0
            for sim in 1:num_simulations
                o_sample = BootstrapParticleFilter.sample_observation_from_particles(particles, weights, u_vec, K, A, p_a, intrusion_dist, no_intrusion_dist)
                new_particles, new_weights = BootstrapParticleFilter.particle_filter_update(particles, weights, u_vec, o_sample, K, A, p_a, intrusion_dist, no_intrusion_dist)
                sim_cost = simulate_rollout(new_particles, new_weights, component_k, partial_control_vec, K, A, p_a, alpha, rollout_horizon, threshold, intrusion_dist, no_intrusion_dist, eta)
                rollout_cost += sim_cost
            end
            rollout_cost /= num_simulations
            return immediate_cost + alpha * rollout_cost + 
                   alpha^(rollout_horizon + 1) * terminal_cost(particles, weights, component_k, partial_control_vec, K, threshold, eta)
        else
            return immediate_cost + alpha * terminal_cost(particles, weights, component_k, partial_control_vec, K, threshold, eta)
        end
    end
    
    future_cost = 0.0
    
    sampled_observations = BootstrapParticleFilter.sample_observations_from_particles(particles, weights, u_vec, K, N, A, p_a, intrusion_dist, no_intrusion_dist)
    
    @inbounds for o_vec in sampled_observations
        new_particles, new_weights = BootstrapParticleFilter.particle_filter_update(particles, weights, u_vec, o_vec, K, A, p_a, intrusion_dist, no_intrusion_dist)
        
        if sum(new_weights) > PROB_THRESHOLD
            u_future = rollout_policy_multiagent_internal(new_particles, new_weights, K, A, p_a, 
                                             alpha, lookahead_horizon - 1, rollout_horizon, num_simulations, N, threshold, 
                                             intrusion_dist, no_intrusion_dist, eta)
            
            q_future = compute_q_value(new_particles, new_weights, u_future, 1, Vector{Int}(), K, 
                                     A, p_a, alpha, 
                                     lookahead_horizon - 1, rollout_horizon, num_simulations, N, threshold, 
                                     intrusion_dist, no_intrusion_dist, eta)
            
            obs_prob = compute_observation_probability(o_vec, new_particles, new_weights, K, intrusion_dist, no_intrusion_dist)
            future_cost += obs_prob * alpha * q_future
        end
    end
    
    return immediate_cost + future_cost
end

function rollout_policy_multiagent_internal(particles, weights, K, A, p_a, 
                                alpha, lookahead_horizon, rollout_horizon, num_simulations, 
                                N, threshold, intrusion_dist, 
                                no_intrusion_dist, eta)
    """
    Internal multiagent rollout that reuses observation distributions for efficiency
    Uses parallel component optimization where each component assumes others follow base policy
    """
    base_control_vec = get_base_policy_vector(particles, weights, K, threshold)
    
    optimal_control_vec = Vector{Int}(undef, K)
    
    Threads.@threads for k in 1:K
        best_component_value = base_control_vec[k]
        best_q_value = Inf
        
        @inbounds for u_k in [0, 1]
            test_control_vec = copy(base_control_vec)
            test_control_vec[k] = u_k
            
            u_vec = NTuple{K,Int}(test_control_vec)
            
            q_value = compute_q_value(particles, weights, u_vec, k, base_control_vec, K,
                                                A, p_a, alpha, lookahead_horizon, rollout_horizon, 
                                                num_simulations, N, threshold, intrusion_dist, no_intrusion_dist, eta)
            
            if q_value < best_q_value
                best_q_value = q_value
                best_component_value = u_k
            end
        end
        
        optimal_control_vec[k] = best_component_value
    end
    
    return NTuple{K,Int}(optimal_control_vec)
end

function simulate_rollout(particles, weights, component_k, 
                                   partial_control_vec, K, A, p_a, alpha, rollout_horizon, 
                                   threshold, intrusion_dist, 
                                   no_intrusion_dist, eta)
    """
    Simulates a rollout using the multiagent base policy (threshold policy)
    """
    total_cost = 0.0
    current_particles = copy(particles)
    current_weights = copy(weights)
    
    for h in 1:rollout_horizon
        u_rollout = base_policy(current_particles, current_weights, K, threshold)
        
        immediate_cost = compute_particle_expected_cost(current_particles, current_weights, u_rollout, eta, K)
        total_cost += alpha^(h-1) * immediate_cost
        
        o_sample = BootstrapParticleFilter.sample_observation_from_particles(current_particles, current_weights, u_rollout, K, A, p_a, intrusion_dist, no_intrusion_dist)
        current_particles, current_weights = BootstrapParticleFilter.particle_filter_update(current_particles, current_weights, u_rollout, o_sample, K, A, p_a, intrusion_dist, no_intrusion_dist)
        
        if sum(current_weights) <= PROB_THRESHOLD
            break
        end
    end
    
    return total_cost
end

function terminal_cost(particles, weights, component_k, 
                                partial_control_vec, K, 
                                threshold, eta)
    """
    Computes terminal cost using multiagent base policy (threshold policy)
    """
    u_terminal = base_policy(particles, weights, K, threshold)
    return compute_particle_expected_cost(particles, weights, u_terminal, eta, K)
end

function base_policy(particles, weights, K, threshold)
    """
    Threshold-based base policy that sets control u=1 for each component 
    if its belief of being compromised is greater than the given threshold
    """
    M = size(particles, 2)
    
    component_beliefs = zeros(K)
    
    @inbounds for i in 1:M
        state_vec = NTuple{K,Int}(particles[:, i])
        weight = weights[i]
        
        for k in 1:K
            if state_vec[k] == 1
                component_beliefs[k] += weight
            end
        end
    end
    
    control_vec = Vector{Int}(undef, K)
    @inbounds for k in 1:K
        control_vec[k] = component_beliefs[k] > threshold ? 1 : 0
    end
    
    return NTuple{K,Int}(control_vec)
end

function compute_observation_probability(o_vec, particles, 
                                        weights, K,
                                        intrusion_dist, 
                                        no_intrusion_dist)
    """
    Compute the probability of an observation vector using the factorized structure
    """
    M = size(particles, 2)
    total_prob = 0.0
    
    @inbounds for i in 1:M
        state_vec = NTuple{K,Int}(particles[:, i])
        
        obs_likelihood = RecoveryPOMDP.compute_observation_likelihood(o_vec, state_vec, intrusion_dist, no_intrusion_dist)
        total_prob += weights[i] * obs_likelihood
    end
    
    return total_prob
end

function run_rollout_simulation(initial_state_vec, K, n,
                                  A, p_a,
                                  alpha, lookahead_horizon, rollout_horizon,
                                  num_simulations, N, M, T, eval_samples, threshold, eta)
    """
    Monte Carlo rollout simulation for T time steps using component-wise observation sampling
    
    Args:
    - initial_state_vec: Initial state vector (deterministic)
    - K: Number of components
    - n: Maximum observation value per component
    - A: Transition matrix
    - p_a: Transition probability
    - alpha: Discount factor
    - lookahead_horizon: Lookahead horizon
    - rollout_horizon: Rollout horizon
    - num_simulations: Number of Monte Carlo simulations
    - N: Number of sample observations for lookahead
    - M: Number of particles
    - T: Number of time steps to simulate
    - eval_samples: Number of evaluation samples to run
    - threshold: Threshold for base policy decision making
    - eta: Cost parameter
    
    Returns:
    - Average total cost across all evaluation samples
    """
    
    intrusion_dist, no_intrusion_dist = RecoveryPOMDP.generate_component_observation_distributions(n)
    total_costs = Vector{Float64}(undef, eval_samples)
    
    @inbounds for sample in 1:eval_samples
        # Create particles from deterministic initial state
        particles = Matrix{Int}(undef, K, M)
        @inbounds for i in 1:M
            particles[:, i] = collect(initial_state_vec)
        end
        weights = fill(1.0 / M, M)
        
        true_state_vec = collect(initial_state_vec)
        
        total_cost = 0.0
        alpha_power = 1.0
        
        @inbounds for t in 1:T
            u_t_vec = rollout_policy(particles, weights, K, n, A, p_a,
                                   alpha, lookahead_horizon, rollout_horizon, num_simulations, N, threshold, eta)
            
            cost_t = RecoveryPOMDP.compute_cost(Tuple(true_state_vec), u_t_vec, eta)
            total_cost += alpha_power * cost_t
            alpha_power *= alpha
            
            if t < T
                o_t = RecoveryPOMDP.sample_observation_vector(Tuple(true_state_vec), intrusion_dist, no_intrusion_dist)
                
                true_state_vec = RecoveryPOMDP.sample_next_state_vector(Tuple(true_state_vec), u_t_vec, p_a, A)
                
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

function get_base_policy_vector(particles, weights, K, threshold)
    """
    Get the base policy control vector (threshold policy)
    """
    M = size(particles, 2)
    
    component_beliefs = zeros(K)
    
    @inbounds for i in 1:M
        state_vec = NTuple{K,Int}(particles[:, i])
        weight = weights[i]
        
        for k in 1:K
            if state_vec[k] == 1
                component_beliefs[k] += weight
            end
        end
    end
    
    control_vec = Vector{Int}(undef, K)
    @inbounds for k in 1:K
        control_vec[k] = component_beliefs[k] > threshold ? 1 : 0
    end
    
    return control_vec
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
