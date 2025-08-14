const PROB_THRESHOLD = 1e-10

function rollout_policy(particles::Matrix{Int}, weights::Vector{Float64}, U::Vector{Int}, 
                          X::Vector{Int}, K::Int, n::Int, x_to_vec::Dict{Int,NTuple{D,Int}},
                          u_to_vec::Dict{Int,NTuple{D,Int}}, vec_to_x::Dict{NTuple{D,Int},Int},
                          vec_to_u::Dict{NTuple{D,Int},Int}, A::Matrix{Int}, p_a::Float64, C::Matrix{Float64}, alpha::Float64, 
                          lookahead_horizon::Int, rollout_horizon::Int, num_simulations::Int, N::Int, threshold::Float64)::Int where D
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
    - U: Control space
    - X: State space  
    - K: Number of components
    - n: Maximum observation value per component
    - x_to_vec: State index to vector mapping
    - u_to_vec: Control index to vector mapping
    - vec_to_x: Vector to state index mapping
    - vec_to_u: Vector to control index mapping
    - A: Transition matrix
    - p_a: Transition probability
    - C: Cost matrix
    - alpha: Discount factor
    - lookahead_horizon: Lookahead horizon
    - rollout_horizon: Rollout horizon
    - num_simulations: Number of Monte Carlo simulations
    - N: Number of sample observations for lookahead
    - threshold: Threshold for base policy
    
    Returns:
    - Rollout control u
    """
    
    intrusion_dist, no_intrusion_dist = RecoveryPOMDP.generate_component_observation_distributions(n)
    
    base_control_vec = get_base_policy_vector(particles, weights, K, x_to_vec, u_to_vec, vec_to_u, threshold)
    
    optimal_control_vec = Vector{Int}(undef, K)
    
    Threads.@threads for k in 1:K
        best_component_value = base_control_vec[k]
        best_q_value = Inf
        
        @inbounds for u_k in [0, 1]
            test_control_vec = copy(base_control_vec)
            test_control_vec[k] = u_k
            
            control_tuple = NTuple{K,Int}(test_control_vec)
            u = vec_to_u[control_tuple]
            
            q_value = compute_q_value(particles, weights, u, k, base_control_vec, U, X, K, x_to_vec, u_to_vec, vec_to_x, vec_to_u,
                                                A, p_a, C, alpha, lookahead_horizon, rollout_horizon, 
                                                num_simulations, N, threshold, intrusion_dist, no_intrusion_dist)
            
            if q_value < best_q_value
                best_q_value = q_value
                best_component_value = u_k
            end
        end
        
        optimal_control_vec[k] = best_component_value
    end
    
    control_tuple = NTuple{K,Int}(optimal_control_vec)
    return vec_to_u[control_tuple]
end

function compute_q_value(particles::Matrix{Int}, weights::Vector{Float64}, u::Int, 
                                   component_k::Int, partial_control_vec::Vector{Int},
                                   U::Vector{Int}, X::Vector{Int}, K::Int, x_to_vec::Dict{Int,NTuple{D,Int}},
                                   u_to_vec::Dict{Int,NTuple{D,Int}}, vec_to_x::Dict{NTuple{D,Int},Int},
                                   vec_to_u::Dict{NTuple{D,Int},Int}, A::Matrix{Int}, p_a::Float64, C::Matrix{Float64}, alpha::Float64, 
                                   lookahead_horizon::Int, rollout_horizon::Int, num_simulations::Int, N::Int, threshold::Float64,
                                   intrusion_dist::Vector{Float64}, no_intrusion_dist::Vector{Float64})::Float64 where D
    """
    Computes Q-value for multiagent rollout where we're optimizing component k
    and assuming remaining components follow base policy
    """
    
    immediate_cost = BootstrapParticleFilter.compute_particle_expected_cost(particles, weights, u, C, X)
    
    if lookahead_horizon == 0
        if rollout_horizon > 0
            rollout_cost = 0.0
            for sim in 1:num_simulations
                o_sample = BootstrapParticleFilter.sample_observation_from_particles(particles, weights, u, K, X, x_to_vec, u_to_vec, vec_to_x, A, p_a, intrusion_dist, no_intrusion_dist)
                new_particles, new_weights = BootstrapParticleFilter.particle_filter_update(particles, weights, u, o_sample, K, X, x_to_vec, u_to_vec, vec_to_x, A, p_a, intrusion_dist, no_intrusion_dist)
                sim_cost = simulate_rollout(new_particles, new_weights, component_k, partial_control_vec, C, X, K, x_to_vec, u_to_vec, vec_to_x, vec_to_u, A, p_a, alpha, rollout_horizon, threshold, intrusion_dist, no_intrusion_dist)
                rollout_cost += sim_cost
            end
            rollout_cost /= num_simulations
            return immediate_cost + alpha * rollout_cost + 
                   alpha^(rollout_horizon + 1) * terminal_cost(particles, weights, component_k, partial_control_vec, C, X, K, x_to_vec, u_to_vec, vec_to_u, threshold)
        else
            return immediate_cost + alpha * terminal_cost(particles, weights, component_k, partial_control_vec, C, X, K, x_to_vec, u_to_vec, vec_to_u, threshold)
        end
    end
    
    future_cost = 0.0
    
    sampled_observations = BootstrapParticleFilter.sample_observations_from_particles(particles, weights, u, K, N, X, x_to_vec, u_to_vec, vec_to_x, A, p_a, intrusion_dist, no_intrusion_dist)
    
    @inbounds for o_vec in sampled_observations
        new_particles, new_weights = BootstrapParticleFilter.particle_filter_update(particles, weights, u, o_vec, K, X, x_to_vec, u_to_vec, vec_to_x, A, p_a, intrusion_dist, no_intrusion_dist)
        
        if sum(new_weights) > PROB_THRESHOLD
            u_future = rollout_policy_multiagent_internal(new_particles, new_weights, U, X, K, x_to_vec, u_to_vec, vec_to_x, vec_to_u, A, p_a, C, 
                                             alpha, lookahead_horizon - 1, rollout_horizon, num_simulations, N, threshold, 
                                             intrusion_dist, no_intrusion_dist)
            
            q_future = compute_q_value(new_particles, new_weights, u_future, 1, Vector{Int}(), U, X, K, 
                                     x_to_vec, u_to_vec, vec_to_x, vec_to_u, A, p_a, C, alpha, 
                                     lookahead_horizon - 1, rollout_horizon, num_simulations, N, threshold, 
                                     intrusion_dist, no_intrusion_dist)
            
            obs_prob = compute_observation_probability(o_vec, new_particles, new_weights, K, x_to_vec, intrusion_dist, no_intrusion_dist)
            future_cost += obs_prob * alpha * q_future
        end
    end
    
    return immediate_cost + future_cost
end

function rollout_policy_multiagent_internal(particles::Matrix{Int}, weights::Vector{Float64}, U::Vector{Int}, 
                                X::Vector{Int}, K::Int, x_to_vec::Dict{Int,NTuple{D,Int}},
                                u_to_vec::Dict{Int,NTuple{D,Int}}, vec_to_x::Dict{NTuple{D,Int},Int},
                                vec_to_u::Dict{NTuple{D,Int},Int}, A::Matrix{Int}, p_a::Float64, C::Matrix{Float64}, 
                                alpha::Float64, lookahead_horizon::Int, rollout_horizon::Int, num_simulations::Int, 
                                N::Int, threshold::Float64, intrusion_dist::Vector{Float64}, 
                                no_intrusion_dist::Vector{Float64})::Int where D
    """
    Internal multiagent rollout that reuses observation distributions for efficiency
    Uses parallel component optimization where each component assumes others follow base policy
    """
    base_control_vec = get_base_policy_vector(particles, weights, K, x_to_vec, u_to_vec, vec_to_u, threshold)
    
    optimal_control_vec = Vector{Int}(undef, K)
    
    Threads.@threads for k in 1:K
        best_component_value = base_control_vec[k]
        best_q_value = Inf
        
        @inbounds for u_k in [0, 1]
            test_control_vec = copy(base_control_vec)
            test_control_vec[k] = u_k
            
            control_tuple = NTuple{K,Int}(test_control_vec)
            u = vec_to_u[control_tuple]
            
            q_value = compute_q_value(particles, weights, u, k, base_control_vec, U, X, K, x_to_vec, u_to_vec, vec_to_x, vec_to_u,
                                                A, p_a, C, alpha, lookahead_horizon, rollout_horizon, 
                                                num_simulations, N, threshold, intrusion_dist, no_intrusion_dist)
            
            if q_value < best_q_value
                best_q_value = q_value
                best_component_value = u_k
            end
        end
        
        optimal_control_vec[k] = best_component_value
    end
    
    control_tuple = NTuple{K,Int}(optimal_control_vec)
    return vec_to_u[control_tuple]
end

function simulate_rollout(particles::Matrix{Int}, weights::Vector{Float64}, component_k::Int, 
                                   partial_control_vec::Vector{Int}, C::Matrix{Float64}, X::Vector{Int}, K::Int, 
                                   x_to_vec::Dict{Int,NTuple{D,Int}}, u_to_vec::Dict{Int,NTuple{D,Int}}, 
                                   vec_to_x::Dict{NTuple{D,Int},Int}, vec_to_u::Dict{NTuple{D,Int},Int},
                                   A::Matrix{Int}, p_a::Float64, alpha::Float64, rollout_horizon::Int, 
                                   threshold::Float64, intrusion_dist::Vector{Float64}, 
                                   no_intrusion_dist::Vector{Float64})::Float64 where D
    """
    Simulates a rollout using the multiagent base policy (threshold policy)
    """
    total_cost = 0.0
    current_particles = copy(particles)
    current_weights = copy(weights)
    
    for h in 1:rollout_horizon
        u_rollout = base_policy(current_particles, current_weights, K, x_to_vec, 
                                                     u_to_vec, vec_to_u, threshold)
        
        immediate_cost = BootstrapParticleFilter.compute_particle_expected_cost(current_particles, current_weights, u_rollout, C, X)
        total_cost += alpha^(h-1) * immediate_cost
        
        o_sample = BootstrapParticleFilter.sample_observation_from_particles(current_particles, current_weights, u_rollout, K, X, x_to_vec, u_to_vec, vec_to_x, A, p_a, intrusion_dist, no_intrusion_dist)
        current_particles, current_weights = BootstrapParticleFilter.particle_filter_update(current_particles, current_weights, u_rollout, o_sample, K, X, x_to_vec, u_to_vec, vec_to_x, A, p_a, intrusion_dist, no_intrusion_dist)
        
        if sum(current_weights) <= PROB_THRESHOLD
            break
        end
    end
    
    return total_cost
end

function terminal_cost(particles::Matrix{Int}, weights::Vector{Float64}, component_k::Int, 
                                partial_control_vec::Vector{Int}, C::Matrix{Float64}, X::Vector{Int}, K::Int, 
                                x_to_vec::Dict{Int,NTuple{D,Int}}, u_to_vec::Dict{Int,NTuple{D,Int}}, 
                                vec_to_u::Dict{NTuple{D,Int},Int}, threshold::Float64)::Float64 where D
    """
    Computes terminal cost using multiagent base policy (threshold policy)
    """
    u_terminal = base_policy(particles, weights, K, x_to_vec, u_to_vec, vec_to_u, threshold)
    return BootstrapParticleFilter.compute_particle_expected_cost(particles, weights, u_terminal, C, X)
end

function base_policy(particles::Matrix{Int}, weights::Vector{Float64}, K::Int, 
                    x_to_vec::Dict{Int,NTuple{D,Int}}, u_to_vec::Dict{Int,NTuple{D,Int}}, 
                    vec_to_u::Dict{NTuple{D,Int},Int}, threshold::Float64)::Int where D
    """
    Threshold-based base policy that sets control u=1 for each component 
    if its belief of being compromised is greater than the given threshold
    """
    M = size(particles, 2)
    
    component_beliefs = zeros(K)
    
    @inbounds for i in 1:M
        state = particles[1, i]
        state_vec = x_to_vec[state]
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
    
    control_tuple = NTuple{K,Int}(control_vec)
    return vec_to_u[control_tuple]
end

function compute_observation_probability(o_vec::NTuple{D,Int}, particles::Matrix{Int}, 
                                        weights::Vector{Float64}, K::Int,
                                        x_to_vec::Dict{Int,NTuple{D,Int}},
                                        intrusion_dist::Vector{Float64}, 
                                        no_intrusion_dist::Vector{Float64})::Float64 where D
    """
    Compute the probability of an observation vector using the factorized structure
    """
    M = size(particles, 2)
    total_prob = 0.0
    
    @inbounds for i in 1:M
        state = particles[1, i]
        state_vec = x_to_vec[state]
        
        obs_likelihood = RecoveryPOMDP.compute_observation_likelihood(o_vec, state_vec, intrusion_dist, no_intrusion_dist)
        total_prob += weights[i] * obs_likelihood
    end
    
    return total_prob
end

function run_rollout_simulation(b_initial::Vector{Float64}, U::Vector{Int}, X::Vector{Int}, K::Int, n::Int,
                                  x_to_vec::Dict{Int,NTuple{D,Int}}, u_to_vec::Dict{Int,NTuple{D,Int}}, 
                                  vec_to_x::Dict{NTuple{D,Int},Int}, vec_to_u::Dict{NTuple{D,Int},Int}, A::Matrix{Int}, p_a::Float64, C::Matrix{Float64},
                                  alpha::Float64, lookahead_horizon::Int, rollout_horizon::Int,
                                  num_simulations::Int, N::Int, M::Int, T::Int, eval_samples::Int, threshold::Float64)::Float64 where D
    """
    Monte Carlo rollout simulation for T time steps using component-wise observation sampling
    
    Args:
    - b_initial: Initial belief state
    - U: Control space
    - X: State space  
    - K: Number of components
    - n: Maximum observation value per component
    - x_to_vec: State index to vector mapping
    - u_to_vec: Control index to vector mapping
    - vec_to_x: Vector to state index mapping
    - vec_to_u: Vector to control index mapping
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
        particles, weights = BootstrapParticleFilter.belief_to_particles(b_initial, X, M)
        true_state = X[rand(Categorical(b_initial))]
        
        total_cost = 0.0
        alpha_power = 1.0
        
        @inbounds for t in 1:T
            u_t = rollout_policy(particles, weights, U, X, K, n, x_to_vec, u_to_vec, vec_to_x, vec_to_u, A, p_a, C,
                                   alpha, lookahead_horizon, rollout_horizon, num_simulations, N, threshold)
            
            cost_t = C[true_state + 1, u_t + 1]
            total_cost += alpha_power * cost_t
            alpha_power *= alpha
            
            if t < T
                true_state_vec = x_to_vec[true_state]
                o_t = RecoveryPOMDP.sample_observation_vector(true_state_vec, intrusion_dist, no_intrusion_dist)
                
                true_state = RecoveryPOMDP.sample_next_state_from_transition_probs(true_state, u_t, p_a, 
                                                                                  x_to_vec, u_to_vec, vec_to_x, A)
                
                particles, weights = BootstrapParticleFilter.particle_filter_update(particles, weights, 
                                                                           u_t, o_t, K, X, x_to_vec, u_to_vec, vec_to_x, A, p_a, intrusion_dist, no_intrusion_dist)
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

function get_base_policy_vector(particles::Matrix{Int}, weights::Vector{Float64}, K::Int,
                               x_to_vec::Dict{Int,NTuple{D,Int}}, u_to_vec::Dict{Int,NTuple{D,Int}}, 
                               vec_to_u::Dict{NTuple{D,Int},Int}, threshold::Float64)::Vector{Int} where D
    """
    Get the base policy control vector (threshold policy)
    """
    M = size(particles, 2)
    
    component_beliefs = zeros(K)
    
    @inbounds for i in 1:M
        state = particles[1, i]
        state_vec = x_to_vec[state]
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
