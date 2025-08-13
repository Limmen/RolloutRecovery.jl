using Distributions
using Distributions: BetaBinomial

function generate_state_space(K::Int)::Tuple{Vector{Int},Dict{Int,NTuple{K,Int}},Dict{NTuple{K,Int},Int}}
    """
    Generate the state space where each server can be in two states: 0 (healthy) and 1 (compromised)

    Returns:
    - X: Vector of state indices
    - x_to_vec: Dictionary mapping state index to state vector
    - vec_to_x: Dictionary mapping state vector to state index
    """
    num_states = 2^K
    X = collect(0:(num_states-1))
    x_to_vec = Dict{Int,NTuple{K,Int}}()
    vec_to_x = Dict{NTuple{K,Int},Int}()

    sizehint!(x_to_vec, num_states)
    sizehint!(vec_to_x, num_states)

    for (idx, x_vec) in enumerate(Iterators.product(fill((0, 1), K)...))
        x_idx = idx - 1
        x_to_vec[x_idx] = x_vec
        vec_to_x[x_vec] = x_idx
    end

    return X, x_to_vec, vec_to_x
end

function generate_control_space(K::Int)::Tuple{Vector{Int},Dict{Int,NTuple{K,Int}},Dict{NTuple{K,Int},Int},Vector{Vector{Int}}}
    """
    Generate the control space where each server has two actions: 0 (continue) and 1 (stop)

    Returns:
    - U: Vector of control indices
    - u_to_vec: Dictionary mapping control index to control vector
    - vec_to_u: Dictionary mapping control vector to control index
    - U_local: Vector of possible controls for each server
    """
    num_controls = 2^K
    U = collect(0:(num_controls-1))
    u_to_vec = Dict{Int,NTuple{K,Int}}()
    vec_to_u = Dict{NTuple{K,Int},Int}()

    sizehint!(u_to_vec, num_controls)
    sizehint!(vec_to_u, num_controls)

    for (idx, u_vec) in enumerate(Iterators.product(fill((0, 1), K)...))
        u_idx = idx - 1
        u_to_vec[u_idx] = u_vec
        vec_to_u[u_vec] = u_idx
    end

    U_local = [Int[0, 1] for _ in 1:K]

    return U, u_to_vec, vec_to_u, U_local
end

function generate_observation_space(n::Int, K::Int)::Tuple{Vector{Int},Dict{Int,NTuple{K,Int}},Dict{NTuple{K,Int},Int}}
    """
    Generate the observation space (0,...n) for each server i in N.

    Args:
    - n: Maximum observation value
    - K: Number of servers

    Returns:
    - O: Vector of observation indices
    - o_to_vec: Dictionary mapping observation index to observation vector
    - vec_to_o: Dictionary mapping observation vector to observation index
    """
    num_observations = (n + 1)^K
    O = collect(0:(num_observations-1))
    o_to_vec = Dict{Int,NTuple{K,Int}}()
    vec_to_o = Dict{NTuple{K,Int},Int}()

    sizehint!(o_to_vec, num_observations)
    sizehint!(vec_to_o, num_observations)

    obs_range = ntuple(_ -> 0:n, K)
    for (idx, o_vec) in enumerate(Iterators.product(obs_range...))
        o_idx = idx - 1
        o_to_vec[o_idx] = o_vec
        vec_to_o[o_vec] = o_idx
    end

    return O, o_to_vec, vec_to_o
end

function initial_state(K::Int, vec_to_x::Dict{NTuple{N,Int},Int}) where N
    """
    Get the initial state index corresponding to all servers being healthy (0)

    Args:    
    - K: Number of servers
    - vec_to_x: Dictionary mapping state vector to state index

    Returns:
    - Initial state index where all servers are healthy
    """
    initial_vec = ntuple(_ -> 0, K)
    return vec_to_x[initial_vec]
end

function initial_belief(K::Int, X::Vector{Int}, vec_to_x::Dict{NTuple{N,Int},Int})::Vector{Float64} where N
    """
    Generate the initial belief vector with probability 1 on the initial state

    Args:
    - K: Number of servers
    - X: Vector of state indices
    - vec_to_x: Dictionary mapping state vector to state index

    Returns:
    - Initial belief vector
    """
    b0 = zeros(Float64, length(X))
    x0 = initial_state(K, vec_to_x)
    b0[x0+1] = 1.0
    return b0
end

function generate_erdos_renyi_graph(K::Int, p_c::Float64)::Matrix{Int}
    """
    Generate the adjacency matrix of a random Erdös-Rényi graph

    Args:
    - K: Number of nodes/servers
    - p_c: Probability of an edge between any two nodes

    Returns:
    - Adjacency matrix of the random graph
    """
    adjacency_matrix = zeros(Int, K, K)

    @inbounds for i in 1:K
        for j in (i+1):K
            if rand() < p_c
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1
            end
        end
    end

    return adjacency_matrix
end

function cost_function(x::Int, u::Int, x_to_vec::Dict{Int,NTuple{N,Int}},
    u_to_vec::Dict{Int,NTuple{N,Int}}, eta::Float64)::Float64 where N
    """
    Compute the cost c(x,u) combining compromise and response costs

    Args:
    - x: State index
    - u: Control index
    - x_to_vec: Dictionary mapping state index to state vector
    - u_to_vec: Dictionary mapping control index to control vector  
    - eta: Weight parameter for compromised costs

    Returns:
    - Total weighted cost
    """
    x_vec = x_to_vec[x]
    u_vec = u_to_vec[u]

    compromised_costs = 0.0
    response_costs = 0.0

    @inbounds for i in 1:N
        compromised_costs += 2 * x_vec[i] * (1 - u_vec[i])
        response_costs += u_vec[i]
    end

    return eta * compromised_costs + response_costs
end

function generate_cost_matrix(X::Vector{Int}, U::Vector{Int},
    x_to_vec::Dict{Int,NTuple{N,Int}},
    u_to_vec::Dict{Int,NTuple{N,Int}},
    eta::Float64)::Matrix{Float64} where N
    """
    Generate a |X|×|U| cost matrix
    """
    cost_matrix = Matrix{Float64}(undef, length(X), length(U))

    @inbounds for (i, x) in enumerate(X)
        for (j, u) in enumerate(U)
            cost_matrix[i, j] = cost_function(x, u, x_to_vec, u_to_vec, eta)
        end
    end

    return cost_matrix
end

@inline function count_compromised_neighbors(i::Int, A::Matrix{Int}, x_vec::NTuple{N,Int})::Int where N
    """
    Count the number of compromised neighbors
    """
    count = 0
    @inbounds for j in 1:N
        count += A[i, j] * x_vec[j]
    end
    return count
end

@inline function local_transition_probability(x_prime::Int, x::Int, u::Int, p_a::Float64,
    num_compromised_neighbors::Int)::Float64
    """
    Transition function of a single node, P(x_prime_i | x_i, u_i)
    """
    if u == 1 && x_prime == 0
        return 1.0
    elseif u == 0
        if x == 1 && x_prime == 1
            return 1.0
        elseif x == 0
            compromise_probability = min(1.0, p_a * (num_compromised_neighbors + 1))
            if x_prime == 0
                return 1.0 - compromise_probability
            elseif x_prime == 1
                return compromise_probability
            end
        end
    end
    return 0.0
end

function transition_probability(x_prime::Int, x::Int, u::Int, K::Int, p_a::Float64,
    x_to_vec::Dict{Int,NTuple{N,Int}},
    u_to_vec::Dict{Int,NTuple{N,Int}},
    A::Matrix{Int})::Float64 where N
    """
    Compute P(x_prime | x,u)
    """
    x_vec = x_to_vec[x]
    x_prime_vec = x_to_vec[x_prime]
    u_vec = u_to_vec[u]

    probability = 1.0
    @inbounds for i in 1:N
        num_neighbors = count_compromised_neighbors(i, A, x_vec)
        local_prob = local_transition_probability(x_prime_vec[i], x_vec[i], u_vec[i], p_a, num_neighbors)
        probability *= local_prob
        if probability == 0.0
            break
        end
    end

    return probability
end

function generate_transition_tensor(p_a::Float64, X::Vector{Int}, U::Vector{Int},
    x_to_vec::Dict{Int,NTuple{N,Int}},
    u_to_vec::Dict{Int,NTuple{N,Int}},
    K::Int, A::Matrix{Int})::Array{Float64,3} where N
    """
    Generate a |U|×|X|×|X| transition tensor
    """
    num_u, num_x = length(U), length(X)
    P = Array{Float64,3}(undef, num_u, num_x, num_x)

    @inbounds for (u_idx, u) in enumerate(U)
        for (x_idx, x) in enumerate(X)
            row_sum = 0.0
            for (x_prime_idx, x_prime) in enumerate(X)
                prob = transition_probability(x_prime, x, u, K, p_a, x_to_vec, u_to_vec, A)
                P[u_idx, x_idx, x_prime_idx] = prob
                row_sum += prob
            end
            @assert isapprox(row_sum, 1.0, rtol=0.01) "Transition probabilities do not sum to 1 for state $x, action $u"
        end
    end

    return P
end

function generate_observation_tensor(n::Int, X::Vector{Int}, K::Int,
    x_to_vec::Dict{Int,NTuple{N,Int}},
    o_to_vec::Dict{Int,NTuple{N,Int}},
    O::Vector{Int})::Matrix{Float64} where N
    """
    Generate a |X|×|O| tensor of observation probabilities

    Args:
    - n: Maximum observation value
    - X: Vector of state indices
    - K: Number of servers
    - x_to_vec: Dictionary mapping state index to state vector
    - o_to_vec: Dictionary mapping observation index to observation vector
    - O: Vector of observation indices

    Returns:
    - Z: Observation probability tensor
    """
    intrusion_dist = Vector{Float64}(undef, n + 1)
    no_intrusion_dist = Vector{Float64}(undef, n + 1)

    intrusion_rv = BetaBinomial(n, 1, 0.7)
    no_intrusion_rv = BetaBinomial(n, 0.7, 3)

    @inbounds for i in 0:n
        intrusion_dist[i+1] = pdf(intrusion_rv, i)
        no_intrusion_dist[i+1] = pdf(no_intrusion_rv, i)
    end

    num_x, num_o = length(X), length(O)
    Z = Matrix{Float64}(undef, num_x, num_o)

    @inbounds for (x_idx, x) in enumerate(X)
        x_vec = x_to_vec[x]
        row_sum = 0.0

        for (o_idx, o) in enumerate(O)
            o_vec = o_to_vec[o]
            probability = 1.0

            for i in 1:N
                if x_vec[i] == 0
                    probability *= no_intrusion_dist[o_vec[i]+1]
                else
                    probability *= intrusion_dist[o_vec[i]+1]
                end
            end

            Z[x_idx, o_idx] = probability
            row_sum += probability
        end

        @assert isapprox(row_sum, 1.0, rtol=0.01) "Observation probabilities do not sum to 1 for state $x"
    end

    return Z
end