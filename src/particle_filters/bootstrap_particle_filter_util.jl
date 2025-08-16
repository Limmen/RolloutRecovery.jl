const PROB_THRESHOLD = 1e-10

function particle_filter_update(particles::Matrix{Int}, weights::Vector{Float64}, 
                               u_vec::NTuple{D,Int}, o_vec::NTuple{D,Int}, K::Int,
                               A::Matrix{Int}, p_a::Float64, intrusion_dist::Vector{Float64}, 
                               no_intrusion_dist::Vector{Float64})::Tuple{Matrix{Int}, Vector{Float64}} where D
    """
    Bootstrap particle filter update step using component-wise observation and state sampling
    """
    
    M = size(particles, 2)
    new_particles = similar(particles)
    new_weights = Vector{Float64}(undef, M)
    
    @inbounds for i in 1:M
        current_state_vec = NTuple{D,Int}(particles[:, i])
        current_weight = weights[i]
        
        next_state_vec = RecoveryPOMDP.sample_next_state_vector(current_state_vec, u_vec, p_a, A)
        new_particles[:, i] = collect(next_state_vec)
        
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

function belief_to_particles(b::Vector{Float64}, M::Int, K::Int)::Tuple{Matrix{Int}, Vector{Float64}}
    """
    Convert belief vector to particle representation using state vectors
    """
    
    particles = Matrix{Int}(undef, K, M)
    weights = fill(1.0 / M, M)
    
    belief_categorical = Categorical(b)
    @inbounds for i in 1:M
        state_idx = rand(belief_categorical)
        # Convert state index to state vector directly
        state_vec = RecoveryPOMDP.state_index_to_vector(state_idx, K)
        particles[:, i] = collect(state_vec)
    end
    
    return particles, weights
end

function sample_observations_from_particles(particles::Matrix{Int}, weights::Vector{Float64}, 
                                           u_vec::NTuple{D,Int}, K::Int, N::Int,
                                           A::Matrix{Int}, p_a::Float64, intrusion_dist::Vector{Float64}, 
                                           no_intrusion_dist::Vector{Float64})::Vector{NTuple{D,Int}} where D
    """
    Sample N observation vectors from the particle distribution using component-wise sampling
    """
    
    observations = Vector{NTuple{D,Int}}(undef, N)
    
    @inbounds for i in 1:N
        particle_idx = sample_particle_index(weights)
        current_state_vec = NTuple{D,Int}(particles[:, particle_idx])
        
        next_state_vec = RecoveryPOMDP.sample_next_state_vector(current_state_vec, u_vec, p_a, A)
        
        observations[i] = RecoveryPOMDP.sample_observation_vector(next_state_vec, intrusion_dist, no_intrusion_dist)
    end
    
    return observations
end

function sample_observation_from_particles(particles::Matrix{Int}, weights::Vector{Float64}, 
                                         u_vec::NTuple{D,Int}, K::Int,
                                         A::Matrix{Int}, p_a::Float64, intrusion_dist::Vector{Float64}, 
                                         no_intrusion_dist::Vector{Float64})::NTuple{D,Int} where D
    """
    Sample a single observation vector from particle distribution using component-wise sampling
    """
    
    particle_idx = sample_particle_index(weights)
    current_state_vec = NTuple{D,Int}(particles[:, particle_idx])
    
    next_state_vec = RecoveryPOMDP.sample_next_state_vector(current_state_vec, u_vec, p_a, A)
    
    return RecoveryPOMDP.sample_observation_vector(next_state_vec, intrusion_dist, no_intrusion_dist)
end

function compute_particle_expected_cost(particles::Matrix{Int}, weights::Vector{Float64}, 
                                       u_vec::NTuple{D,Int}, C::Matrix{Float64}, K::Int)::Float64 where D
    """
    Compute expected cost using particle representation with state vectors
    """
    
    M = size(particles, 2)
    expected_cost = 0.0
    
    @inbounds for i in 1:M
        state_vec = NTuple{D,Int}(particles[:, i])
        state_idx = RecoveryPOMDP.state_vector_to_index(state_vec, K)
        u_idx = RecoveryPOMDP.control_vector_to_index(u_vec, K)
        cost = C[state_idx, u_idx]
        expected_cost += weights[i] * cost
    end
    
    return expected_cost
end
