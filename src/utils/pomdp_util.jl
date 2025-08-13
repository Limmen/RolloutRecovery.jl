using Distributions
using LinearAlgebra
using Statistics

function belief_operator(z::Int, u::Int, b::Vector{Float64}, X::Vector{Int}, Z::Matrix{Float64}, P::Array{Float64,3})::Vector{Float64}
    """
    Computes b' after observing (b,o)
    """
    n_states = length(X)
    b_prime = Vector{Float64}(undef, n_states)

    @inbounds for i in 1:n_states
        x_prime = X[i]
        b_prime[i] = bayes_filter(x_prime, z, u, b, X, Z, P)
    end

    @assert round(sum(b_prime), digits=2) == 1.0
    return b_prime
end

function bayes_filter(x_prime::Int, z::Int, u::Int, b::Vector{Float64}, X::Vector{Int}, Z::Matrix{Float64}, P::Array{Float64,3})::Float64
    """
    A Bayesian filter to compute b[x_prime] after observing (z,u)
    """
    z_idx = z + 1
    u_idx = u + 1
    x_prime_idx = x_prime + 1
    n_states = length(X)

    norm = 0.0
    @inbounds for i in 1:n_states
        x = X[i]
        x_idx = x + 1
        b_x = b[x_idx]
        P_u_x = view(P, u_idx, x_idx, :)

        @simd for j in 1:n_states
            x_prime_1 = X[j]
            x_prime_1_idx = x_prime_1 + 1
            prob_1 = Z[x_prime_1_idx, z_idx]
            norm += b_x * prob_1 * P_u_x[x_prime_1_idx]
        end
    end

    temp = 0.0
    z_x_prime = Z[x_prime_idx, z_idx]
    @inbounds @simd for i in 1:n_states
        x = X[i]
        x_idx = x + 1
        temp += z_x_prime * P[u_idx, x_idx, x_prime_idx] * b[x_idx]
    end

    b_prime_s_prime = temp / norm
    @assert round(b_prime_s_prime, digits=2) <= 1.0
    return b_prime_s_prime
end

function expected_cost(b::Vector{Float64}, u::Int, C::Matrix{Float64}, X::Vector{Int})::Float64
    """
    Computes E[C[x][u] | b]
    """
    u_idx = u + 1
    n_states = length(X)
    cost = 0.0

    @inbounds @simd for i in 1:n_states
        x = X[i]
        x_idx = x + 1
        cost += C[x_idx, u_idx] * b[x_idx]
    end

    return cost
end

function compute_observation_probability(b::Vector{Float64}, u::Int, o::Int, X::Vector{Int},
    P::Array{Float64,3}, Z::Matrix{Float64})::Float64
    """
    Computes P(o | b, u)
    """
    prob = 0.0
    for (i, x) in enumerate(X)
        for (j, x_prime) in enumerate(X)
            prob += b[i] * P[u+1, x+1, x_prime+1] * Z[x_prime+1, o+1]
        end
    end
    return prob
end

function sample_observation(b::Vector{Float64}, u::Int, O::Vector{Int}, X::Vector{Int},
    P::Array{Float64,3}, Z::Matrix{Float64})::Int
    """
    Samples observation from P(o | b, u)
    """
    obs_probs = zeros(length(O))

    for (k, o) in enumerate(O)
        obs_probs[k] = compute_observation_probability(b, u, o, X, P, Z)
    end

    normalizer = sum(obs_probs)
    if normalizer > 1e-10
        obs_probs ./= normalizer
        return rand(Categorical(obs_probs)) - 1
    else
        return O[1]
    end
end