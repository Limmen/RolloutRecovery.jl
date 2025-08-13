using Distributions

const PROB_THRESHOLD = 1e-10

function rollout_policy(b::Vector{Float64}, U::Vector{Int}, X::Vector{Int}, O::Vector{Int},
    P::Array{Float64,3}, Z::Matrix{Float64}, C::Matrix{Float64},
    x_to_vec::Dict{Int,NTuple{K,Int}}, u_to_vec::Dict{Int,NTuple{K,Int}}, 
    vec_to_u::Dict{NTuple{K,Int},Int},
    alpha::Float64, lookahead_horizon::Int, rollout_horizon::Int,
    num_simulations::Int)::Int where K
    """
    Implements the basic rollout algorithm

    Args:
    - b: Current belief state
    - U: Control space
    - X: State space  
    - O: Observation space
    - P: Transition tensor P[u+1, x+1, x'+1]
    - Z: Observation tensor Z[x+1, o+1]
    - C: Cost matrix C[x+1, u+1]
    - x_to_vec: State index to vector mapping
    - u_to_vec: Control index to vector mapping
    - vec_to_u: Vector to control index mapping
    - alpha: Discount factor
    - lookahead_horizon: Lookahead horizon ℓ
    - rollout_horizon: Rollout horizon m
    - num_simulations: Number of Monte Carlo simulations L

    Returns:
    - Rollout control u
    """

    best_control = U[1]
    best_value = Inf

    @inbounds for u in U
        q_value = compute_q_value(b, u, U, X, O, P, Z, C, x_to_vec, u_to_vec, vec_to_u,
            alpha, lookahead_horizon, rollout_horizon, num_simulations)

        if q_value < best_value
            best_value = q_value
            best_control = u
        end
    end

    return best_control
end

function compute_q_value(b::Vector{Float64}, u::Int, U::Vector{Int}, X::Vector{Int}, O::Vector{Int},
    P::Array{Float64,3}, Z::Matrix{Float64}, C::Matrix{Float64},
    x_to_vec::Dict{Int,NTuple{K,Int}}, u_to_vec::Dict{Int,NTuple{K,Int}}, 
    vec_to_u::Dict{NTuple{K,Int},Int},
    alpha::Float64, lookahead_horizon::Int, rollout_horizon::Int,
    num_simulations::Int)::Float64 where K
    """
    Computes Q-value for control u in belief state b using lookahead optimization
    """

    immediate_cost = POMDPUtil.expected_cost(b, u, C, X)

    if lookahead_horizon == 0
        if rollout_horizon > 0
            rollout_cost = 0.0
            for sim in 1:num_simulations
                sim_cost = simulate_rollout(b, P, Z, C, X, O, x_to_vec, u_to_vec, vec_to_u, U, alpha, rollout_horizon)
                rollout_cost += sim_cost
            end
            rollout_cost /= num_simulations
            return immediate_cost + alpha * rollout_cost + alpha^(rollout_horizon + 1) * terminal_cost(b, C, X, x_to_vec, u_to_vec, vec_to_u, U)
        else
            return immediate_cost + alpha * terminal_cost(b, C, X, x_to_vec, u_to_vec, vec_to_u, U)
        end
    end

    future_cost = 0.0

    @inbounds for o in O
        prob_o = POMDPUtil.compute_observation_probability(b, u, o, X, P, Z)

        if prob_o > PROB_THRESHOLD
            b_next = POMDPUtil.belief_operator(o, u, b, X, Z, P)
            
            min_expected_cost = Inf
            @inbounds for u_next in U
                q_next = compute_q_value(b_next, u_next, U, X, O, P, Z, C, x_to_vec, u_to_vec, vec_to_u,
                                       alpha, lookahead_horizon - 1, rollout_horizon, num_simulations)
                min_expected_cost = min(min_expected_cost, q_next)
            end

            future_cost += prob_o * alpha * min_expected_cost
        end
    end

    return immediate_cost + future_cost
end

function simulate_rollout(b::Vector{Float64}, P::Array{Float64,3}, Z::Matrix{Float64}, 
                         C::Matrix{Float64}, X::Vector{Int}, O::Vector{Int},
                         x_to_vec::Dict{Int,NTuple{K,Int}}, u_to_vec::Dict{Int,NTuple{K,Int}}, 
                         vec_to_u::Dict{NTuple{K,Int},Int}, U::Vector{Int},
                         alpha::Float64, horizon::Int)::Float64 where K
    """
    Simulates rollout following base policy for specified horizon
    """
    
    current_belief = copy(b)
    total_cost = 0.0
    alpha_power = 1.0
    
    @inbounds for t in 1:horizon
        u_t = base_policy(current_belief, X, U, x_to_vec, u_to_vec, vec_to_u)
        cost_t = POMDPUtil.expected_cost(current_belief, u_t, C, X)
        total_cost += alpha_power * cost_t
        alpha_power *= alpha
        
        if t < horizon
            o_t = POMDPUtil.sample_observation(current_belief, u_t, O, X, P, Z)
            current_belief = POMDPUtil.belief_operator(o_t, u_t, current_belief, X, Z, P)
        end
    end
    
    return total_cost
end

function terminal_cost(b::Vector{Float64}, C::Matrix{Float64}, X::Vector{Int},
                      x_to_vec::Dict{Int,NTuple{K,Int}}, u_to_vec::Dict{Int,NTuple{K,Int}}, 
                      vec_to_u::Dict{NTuple{K,Int},Int}, U::Vector{Int})::Float64 where K
    """
    Computes terminal cost as expected cost at the belief state using base policy
    """
    u_terminal = base_policy(b, X, U, x_to_vec, u_to_vec, vec_to_u)
    return POMDPUtil.expected_cost(b, u_terminal, C, X)
end

function base_policy(b::Vector{Float64}, X::Vector{Int}, U::Vector{Int}, 
                    x_to_vec::Dict{Int,NTuple{K,Int}}, u_to_vec::Dict{Int,NTuple{K,Int}}, 
                    vec_to_u::Dict{NTuple{K,Int},Int})::Int where K
    """
    Threshold-based base policy that sets control u=1 for each component 
    if its belief of being compromised is greater than 0.5
    """
    
    # Compute belief probabilities for each component being compromised
    component_beliefs = zeros(K)
    
    @inbounds for i in 1:length(X)
        state = X[i]
        state_vec = x_to_vec[state]
        belief_prob = b[i]
        
        # Add belief probability to compromised belief for each component that is compromised (state=1)
        for k in 1:K
            if state_vec[k] == 1
                component_beliefs[k] += belief_prob
            end
        end
    end
    
    # Create control vector based on threshold policy
    control_vec = Vector{Int}(undef, K)
    @inbounds for k in 1:K
        control_vec[k] = component_beliefs[k] > 0.5 ? 1 : 0
    end
    
    # Convert control vector to control index
    control_tuple = NTuple{K,Int}(control_vec)
    return vec_to_u[control_tuple]
end

function sample_observation_from_state(state::Int, Z::Matrix{Float64}, O::Vector{Int})::Int
    """
    Sample an observation from a given state using the observation tensor
    """
    state_idx = state + 1
    observation_probs = Z[state_idx, :]
    
    return O[rand(Categorical(observation_probs))] 
end

function sample_next_state(current_state::Int, control::Int, P::Array{Float64,3}, X::Vector{Int})::Int
    """
    Sample next state given current state and control using transition tensor
    """
    control_idx = control + 1
    current_state_idx = current_state + 1
    
    transition_probs = P[control_idx, current_state_idx, :]
    
    return X[rand(Categorical(transition_probs))]
end

function run_rollout_simulation(b_initial::Vector{Float64}, U::Vector{Int}, X::Vector{Int}, O::Vector{Int},
                               P::Array{Float64,3}, Z::Matrix{Float64}, C::Matrix{Float64},
                               x_to_vec::Dict{Int,NTuple{K,Int}}, u_to_vec::Dict{Int,NTuple{K,Int}}, 
                               vec_to_u::Dict{NTuple{K,Int},Int},
                               alpha::Float64, lookahead_horizon::Int, rollout_horizon::Int,
                               num_simulations::Int, T::Int, eval_samples::Int)::Float64 where K
    """
    Runs a simulation for T time steps using rollout policy
    
    Args:
    - b_initial: Initial belief state
    - U: Control space
    - X: State space  
    - O: Observation space
    - P: Transition tensor P[u+1, x+1, x'+1]
    - Z: Observation tensor Z[x+1, o+1]
    - C: Cost matrix C[x+1, u+1]
    - x_to_vec: State index to vector mapping
    - u_to_vec: Control index to vector mapping
    - vec_to_u: Vector to control index mapping
    - alpha: Discount factor
    - lookahead_horizon: Lookahead horizon ℓ
    - rollout_horizon: Rollout horizon m
    - num_simulations: Number of Monte Carlo simulations L
    - T: Number of time steps to simulate
    - eval_samples: Number of evaluation samples to run
    
    Returns:
    - Average total cost across all evaluation samples
    """
    
    total_costs = Vector{Float64}(undef, eval_samples)
    
    @inbounds for sample in 1:eval_samples
        current_belief = copy(b_initial)
        true_state = X[rand(Categorical(b_initial))]
        
        total_cost = 0.0
        alpha_power = 1.0
        
        @inbounds for t in 1:T
            u_t = rollout_policy(current_belief, U, X, O, P, Z, C, x_to_vec, u_to_vec, vec_to_u,
                                alpha, lookahead_horizon, rollout_horizon, num_simulations)
            
            cost_t = C[true_state + 1, u_t + 1]
            total_cost += alpha_power * cost_t
            alpha_power *= alpha
            
            if t < T
                o_t = sample_observation_from_state(true_state, Z, O)
                true_state = sample_next_state(true_state, u_t, P, X)
                current_belief = POMDPUtil.belief_operator(o_t, u_t, current_belief, X, Z, P)
            end
        end
        
        total_costs[sample] = total_cost
    end
    
    return sum(total_costs) / eval_samples
end