const PROB_THRESHOLD = 1e-10

function particle_filter_update(particles::Matrix{Int}, weights::Vector{Float64}, 
                               u::Int, o_vec::NTuple{D,Int}, K::Int,
                               X::Vector{Int}, x_to_vec::Dict{Int,NTuple{D,Int}},
                               u_to_vec::Dict{Int,NTuple{D,Int}}, vec_to_x::Dict{NTuple{D,Int},Int},
                               A::Matrix{Int}, p_a::Float64, intrusion_dist::Vector{Float64}, 
                               no_intrusion_dist::Vector{Float64})::Tuple{Matrix{Int}, Vector{Float64}} where D
    """
    Bootstrap particle filter update step using component-wise observation and state sampling
    """
    
    M = size(particles, 2)
    new_particles = similar(particles)
    new_weights = Vector{Float64}(undef, M)
    
    @inbounds for i in 1:M
        current_state = particles[1, i]
        current_weight = weights[i]
        
        next_state = RecoveryPOMDP.sample_next_state_from_transition_probs(current_state, u, p_a, 
                                                                          x_to_vec, u_to_vec, vec_to_x, A)
        new_particles[1, i] = next_state
        
        next_state_vec = x_to_vec[next_state]
        obs_likelihood = RecoveryPOMDP.compute_observation_likelihood(o_vec, next_state_vec, intrusion_dist, no_intrusion_dist)
        new_weights[i] = current_weight * obs_likelihood
    end
    
    weight_sum = sum(new_weights)
    if weight_sum > PROB_THRESHOLD
        new_weights ./= weight_sum
    else
        fill!(new_weights, 1.0 / M)
    end
    
    ess = 1.0 / sum(new_weights.^2)
    if ess < M / 2
        new_particles, new_weights = resample_particles(new_particles, new_weights)
    end
    
    return new_particles, new_weights
end

function resample_particles(particles::Matrix{Int}, weights::Vector{Float64})::Tuple{Matrix{Int}, Vector{Float64}}
    """
    Systematic resampling of particles
    """
    
    M = size(particles, 2)
    new_particles = similar(particles)
    new_weights = fill(1.0 / M, M)
    
    cumsum_weights = cumsum(weights)
    u = rand() / M
    
    @inbounds for i in 1:M
        target = u + (i - 1) / M
        idx = findfirst(w -> w >= target, cumsum_weights)
        if idx === nothing
            idx = M
        end
        new_particles[:, i] = particles[:, idx]
    end
    
    return new_particles, new_weights
end

function sample_particle_index(weights::Vector{Float64})::Int
    """
    Sample a particle index according to weights
    """
    return rand(Categorical(weights))
end

function belief_to_particles(b::Vector{Float64}, X::Vector{Int}, M::Int)::Tuple{Matrix{Int}, Vector{Float64}}
    """
    Convert belief vector to particle representation
    """
    
    particles = Matrix{Int}(undef, 1, M)
    weights = fill(1.0 / M, M)
    
    belief_categorical = Categorical(b)
    @inbounds for i in 1:M
        state_idx = rand(belief_categorical)
        particles[1, i] = X[state_idx]
    end
    
    return particles, weights
end

function sample_observations_from_particles(particles::Matrix{Int}, weights::Vector{Float64}, 
                                           u::Int, K::Int, N::Int,
                                           X::Vector{Int}, x_to_vec::Dict{Int,NTuple{D,Int}},
                                           u_to_vec::Dict{Int,NTuple{D,Int}}, vec_to_x::Dict{NTuple{D,Int},Int},
                                           A::Matrix{Int}, p_a::Float64, intrusion_dist::Vector{Float64}, 
                                           no_intrusion_dist::Vector{Float64})::Vector{NTuple{D,Int}} where D
    """
    Sample N observation vectors from the particle distribution using component-wise sampling
    """
    
    observations = Vector{NTuple{D,Int}}(undef, N)
    
    @inbounds for i in 1:N
        particle_idx = sample_particle_index(weights)
        current_state = particles[1, particle_idx]
        
        next_state = RecoveryPOMDP.sample_next_state_from_transition_probs(current_state, u, p_a, 
                                                                          x_to_vec, u_to_vec, vec_to_x, A)
        next_state_vec = x_to_vec[next_state]
        
        observations[i] = RecoveryPOMDP.sample_observation_vector(next_state_vec, intrusion_dist, no_intrusion_dist)
    end
    
    return observations
end

function sample_observation_from_particles(particles::Matrix{Int}, weights::Vector{Float64}, 
                                         u::Int, K::Int,
                                         X::Vector{Int}, x_to_vec::Dict{Int,NTuple{D,Int}},
                                         u_to_vec::Dict{Int,NTuple{D,Int}}, vec_to_x::Dict{NTuple{D,Int},Int},
                                         A::Matrix{Int}, p_a::Float64, intrusion_dist::Vector{Float64}, 
                                         no_intrusion_dist::Vector{Float64})::NTuple{D,Int} where D
    """
    Sample a single observation vector from particle distribution using component-wise sampling
    """
    
    particle_idx = sample_particle_index(weights)
    current_state = particles[1, particle_idx]
    
    next_state = RecoveryPOMDP.sample_next_state_from_transition_probs(current_state, u, p_a, 
                                                                      x_to_vec, u_to_vec, vec_to_x, A)
    next_state_vec = x_to_vec[next_state]
    
    return RecoveryPOMDP.sample_observation_vector(next_state_vec, intrusion_dist, no_intrusion_dist)
end

function compute_particle_expected_cost(particles::Matrix{Int}, weights::Vector{Float64}, 
                                       u::Int, C::Matrix{Float64}, X::Vector{Int})::Float64
    """
    Compute expected cost using particle representation
    """
    
    M = size(particles, 2)
    expected_cost = 0.0
    
    @inbounds for i in 1:M
        state = particles[1, i]
        state_idx = findfirst(x -> x == state, X)
        cost = C[state_idx, u+1]
        expected_cost += weights[i] * cost
    end
    
    return expected_cost
end
