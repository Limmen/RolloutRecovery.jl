const PROB_THRESHOLD = 1e-10

function rollout_policy(particles::Matrix{Int}, weights::Vector{Float64}, U::Vector{Int}, 
                          X::Vector{Int}, K::Int, n::Int, x_to_vec::Dict{Int,NTuple{D,Int}},
                          u_to_vec::Dict{Int,NTuple{D,Int}}, vec_to_x::Dict{NTuple{D,Int},Int},
                          A::Matrix{Int}, p_a::Float64, C::Matrix{Float64}, alpha::Float64, 
                          lookahead_horizon::Int, rollout_horizon::Int, num_simulations::Int, N::Int)::Int where D
    """
    Implements Monte Carlo rollout algorithm using component-wise observation sampling
    
    Args:
    - particles: Matrix where each column is a particle (state vector)
    - weights: Particle weights
    - U: Control space
    - X: State space  
    - K: Number of components
    - n: Maximum observation value per component
    - x_to_vec: State index to vector mapping
    - P: Transition tensor P[u+1, x+1, x'+1]
    - C: Cost matrix C[x+1, u+1]
    - alpha: Discount factor
    - lookahead_horizon: Lookahead horizon ℓ
    - rollout_horizon: Rollout horizon m
    - num_simulations: Number of Monte Carlo simulations L
    - N: Number of sample observations for lookahead
    
    Returns:
    - Rollout control u
    """
    
    # Generate component distributions once and reuse
    intrusion_dist, no_intrusion_dist = RecoveryPOMDP.generate_component_observation_distributions(n)
    
    best_control = U[1]
    best_value = Inf
    
    @inbounds for u in U
        q_value = compute_q_value(particles, weights, u, U, X, K, x_to_vec, u_to_vec, vec_to_x,
                                    A, p_a, C, alpha, lookahead_horizon, rollout_horizon, 
                                    num_simulations, N, intrusion_dist, no_intrusion_dist)
        
        if q_value < best_value
            best_value = q_value
            best_control = u
        end
    end
    
    return best_control
end

function compute_q_value(particles::Matrix{Int}, weights::Vector{Float64}, u::Int, 
                           U::Vector{Int}, X::Vector{Int}, K::Int, x_to_vec::Dict{Int,NTuple{D,Int}},
                           u_to_vec::Dict{Int,NTuple{D,Int}}, vec_to_x::Dict{NTuple{D,Int},Int},
                           A::Matrix{Int}, p_a::Float64, C::Matrix{Float64}, alpha::Float64, 
                           lookahead_horizon::Int, rollout_horizon::Int, num_simulations::Int, N::Int,
                           intrusion_dist::Vector{Float64}, no_intrusion_dist::Vector{Float64})::Float64 where D
    """
    Computes Q-value using Monte Carlo sampling with component-wise observation sampling
    """
    
    immediate_cost = BootstrapParticleFilter.compute_particle_expected_cost(particles, weights, u, C, X)
    
    if lookahead_horizon == 0
        if rollout_horizon > 0
            rollout_cost = 0.0
            for sim in 1:num_simulations
                sim_cost = simulate_rollout(particles, weights, C, X, K, x_to_vec, u_to_vec, vec_to_x, A, p_a, alpha, rollout_horizon, intrusion_dist, no_intrusion_dist)
                rollout_cost += sim_cost
            end
            rollout_cost /= num_simulations
            return immediate_cost + alpha * rollout_cost + 
                   alpha^(rollout_horizon + 1) * terminal_cost(particles, weights, C, X)
        else
            return immediate_cost + alpha * terminal_cost(particles, weights, C, X)
        end
    end
    
    future_cost = 0.0
    
    sampled_observations = BootstrapParticleFilter.sample_observations_from_particles(particles, weights, u, K, N, X, x_to_vec, u_to_vec, vec_to_x, A, p_a, intrusion_dist, no_intrusion_dist)
    
    @inbounds for o_vec in sampled_observations
        new_particles, new_weights = BootstrapParticleFilter.particle_filter_update(particles, weights, u, o_vec, K, X, x_to_vec, u_to_vec, vec_to_x, A, p_a, intrusion_dist, no_intrusion_dist)
        
        if sum(new_weights) > PROB_THRESHOLD
            min_expected_cost = Inf
            @inbounds for u_next in U
                q_next = compute_q_value(new_particles, new_weights, u_next, U, X, K, x_to_vec, u_to_vec, vec_to_x,
                                          A, p_a, C, alpha, lookahead_horizon - 1, rollout_horizon, 
                                          num_simulations, N, intrusion_dist, no_intrusion_dist)
                min_expected_cost = min(min_expected_cost, q_next)
            end
            
            obs_prob = compute_observation_probability(o_vec, new_particles, new_weights, K, x_to_vec, intrusion_dist, no_intrusion_dist)
            future_cost += obs_prob * alpha * min_expected_cost
        end
    end
    
    return immediate_cost + future_cost
end

function simulate_rollout(particles::Matrix{Int}, weights::Vector{Float64}, 
                            C::Matrix{Float64}, X::Vector{Int}, K::Int, x_to_vec::Dict{Int,NTuple{D,Int}},
                            u_to_vec::Dict{Int,NTuple{D,Int}}, vec_to_x::Dict{NTuple{D,Int},Int},
                            A::Matrix{Int}, p_a::Float64, alpha::Float64, horizon::Int, 
                            intrusion_dist::Vector{Float64}, no_intrusion_dist::Vector{Float64})::Float64 where D
    """
    Monte Carlo rollout simulation using component-wise observation sampling
    """
    
    current_particles = copy(particles)
    current_weights = copy(weights)
    total_cost = 0.0
    alpha_power = 1.0
    
    @inbounds for t in 1:horizon
        u_t = base_policy()
        cost_t = BootstrapParticleFilter.compute_particle_expected_cost(current_particles, current_weights, u_t, C, X)
        total_cost += alpha_power * cost_t
        alpha_power *= alpha
        
        if t < horizon
            o_t = BootstrapParticleFilter.sample_observation_from_particles(current_particles, current_weights, u_t, K, X, x_to_vec, u_to_vec, vec_to_x, A, p_a, intrusion_dist, no_intrusion_dist)
            current_particles, current_weights = BootstrapParticleFilter.particle_filter_update(current_particles, current_weights, 
                                                                       u_t, o_t, K, X, x_to_vec, u_to_vec, vec_to_x, A, p_a, intrusion_dist, no_intrusion_dist)
        end
    end
    
    return total_cost
end

function terminal_cost(particles::Vector{Vector{Int}}, weights::Vector{Float64}, 
                         C::Matrix{Float64}, X::Int)::Float64
    """
    Compute terminal cost using particle representation
    """
    u_terminal = base_policy()
    return BootstrapParticleFilter.compute_particle_expected_cost(particles, weights, u_terminal, C, X)
end

@inline function base_policy()::Int
    """
    Deterministic base policy that always selects control=0
    """
    return 0
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
                                  x_to_vec::Dict{Int,NTuple{D,Int}}, u_to_vec::Dict{NTuple{D,Int},Int}, 
                                  vec_to_x::Dict{NTuple{D,Int},Int}, A::Matrix{Int}, p_a::Float64, C::Matrix{Float64},
                                  alpha::Float64, lookahead_horizon::Int, rollout_horizon::Int,
                                  num_simulations::Int, N::Int, M::Int, T::Int)::NamedTuple{(:controls, :observations, :particles_history, :costs, :total_cost), Tuple{Vector{Int}, Vector{NTuple{D,Int}}, Vector{Tuple{Matrix{Int}, Vector{Float64}}}, Vector{Float64}, Float64}} where D
    """
    Monte Carlo rollout simulation for T time steps using component-wise observation sampling
    
    Args:
    - b_initial: Initial belief state
    - U: Control space
    - X: State space  
    - K: Number of components
    - n: Maximum observation value per component
    - x_to_vec: State index to vector mapping
    - P: Transition tensor P[u+1, x+1, x'+1]
    - C: Cost matrix C[x+1, u+1]
    - alpha: Discount factor
    - lookahead_horizon: Lookahead horizon ℓ
    - rollout_horizon: Rollout horizon m
    - num_simulations: Number of Monte Carlo simulations L
    - N: Number of sample observations for lookahead
    - M: Number of particles
    - T: Number of time steps to simulate
    
    Returns:
    - Named tuple with simulation results
    """
    
    controls = Vector{Int}(undef, T)
    observations = Vector{NTuple{D,Int}}(undef, T)
    particles_history = Vector{Tuple{Matrix{Int}, Vector{Float64}}}(undef, T+1)
    costs = Vector{Float64}(undef, T)
    
    # Generate component distributions once for the entire simulation
    intrusion_dist, no_intrusion_dist = RecoveryPOMDP.generate_component_observation_distributions(n)
    
    # Generate state transition mappings
    u_to_vec, vec_to_x, A, p_a = RecoveryPOMDP.generate_state_transition_mappings(K, n)
    
    current_particles, current_weights = BootstrapParticleFilter.belief_to_particles(b_initial, X, M)
    particles_history[1] = (copy(current_particles), copy(current_weights))
    
    total_cost = 0.0
    alpha_power = 1.0
    
    @inbounds for t in 1:T
        u_t = rollout_policy(current_particles, current_weights, U, X, K, n, x_to_vec, u_to_vec, vec_to_x, A, p_a, C,
                               alpha, lookahead_horizon, rollout_horizon, num_simulations, N)
        controls[t] = u_t
        
        cost_t = BootstrapParticleFilter.compute_particle_expected_cost(current_particles, current_weights, u_t, C, X)
        costs[t] = cost_t
        total_cost += alpha_power * cost_t
        alpha_power *= alpha
        
        o_t = BootstrapParticleFilter.sample_observation_from_particles(current_particles, current_weights, u_t, K, X, x_to_vec, u_to_vec, vec_to_x, A, p_a, intrusion_dist, no_intrusion_dist)
        observations[t] = o_t
        
        if t < T
            current_particles, current_weights = BootstrapParticleFilter.particle_filter_update(current_particles, current_weights, 
                                                                       u_t, o_t, K, X, x_to_vec, u_to_vec, vec_to_x, A, p_a, intrusion_dist, no_intrusion_dist)
            particles_history[t+1] = (copy(current_particles), copy(current_weights))
        end
    end
    
    return (controls=controls, observations=observations, particles_history=particles_history, 
            costs=costs, total_cost=total_cost)
end
