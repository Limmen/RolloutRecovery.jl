using Distributions
using LinearAlgebra
using Statistics

function rollout_policy(b::Vector{Float64}, U::Vector{Int}, X::Vector{Int}, O::Vector{Int}, 
                       P::Array{Float64,3}, Z::Matrix{Float64}, C::Matrix{Float64}, 
                       gamma::Float64, lookahead_horizon::Int, rollout_horizon::Int, 
                       num_simulations::Int)::Int
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
    - gamma: Discount factor
    - lookahead_horizon: Lookahead horizon ℓ
    - rollout_horizon: Rollout horizon m
    - num_simulations: Number of Monte Carlo simulations L
    
    Returns:
    - Rollout control u
    """
    
    best_control = U[1]
    best_value = Inf
    
    for u in U
        q_value = compute_q_value(b, u, U, X, O, P, Z, C, 
                                gamma, lookahead_horizon, rollout_horizon, num_simulations)
        
        if q_value < best_value
            best_value = q_value
            best_control = u
        end
    end
    
    return best_control
end

function compute_q_value(b_k::Vector{Float64}, u_k::Int, U::Vector{Int}, X::Vector{Int}, O::Vector{Int},
                        P::Array{Float64,3}, Z::Matrix{Float64}, C::Matrix{Float64},
                        gamma::Float64, lookahead_horizon::Int, rollout_horizon::Int, 
                        num_simulations::Int)::Float64
    """
    Computes Q-value for control u_k in belief state b_k using lookahead optimization
    """
    
    immediate_cost = POMDPUtil.expected_cost(b_k, u_k, C, X)
    
    future_cost = 0.0
    
    for o in O
        prob_o = POMDPUtil.compute_observation_probability(b_k, u_k, o, X, P, Z)
        
        if prob_o > 1e-10
            b_next = POMDPUtil.belief_operator(o, u_k, b_k, X, Z, P)
            
            min_expected_cost = Inf
            
            for sequence_idx in 1:num_simulations
                expected_cost_seq = simulate_lookahead_sequence(b_next, U, X, O, P, Z, C, 
                                                          gamma, lookahead_horizon, rollout_horizon)
                min_expected_cost = min(min_expected_cost, expected_cost_seq)
            end
            
            future_cost += prob_o * gamma * min_expected_cost
        end
    end
    
    return immediate_cost + future_cost
end

function simulate_lookahead_sequence(b::Vector{Float64}, U::Vector{Int}, X::Vector{Int}, O::Vector{Int},
                                   P::Array{Float64,3}, Z::Matrix{Float64}, C::Matrix{Float64},
                                   gamma::Float64, lookahead_horizon::Int, rollout_horizon::Int)::Float64
    """
    Simulates a sequence following the base policy for lookahead steps, 
    then estimates cost-to-go using rollout
    """
    
    current_belief = copy(b)
    total_cost = 0.0
    
    for j in 1:lookahead_horizon
        u_j = base_policy()
        
        cost_j = POMDPUtil.expected_cost(current_belief, u_j, C, X)
        total_cost += gamma^(j-1) * cost_j
        
        if j < lookahead_horizon
            o_j = POMDPUtil.sample_observation(current_belief, u_j, O, X, P, Z)
            current_belief = POMDPUtil.belief_operator(o_j, u_j, current_belief, X, Z, P)
        end
    end
    
    if rollout_horizon > 0
        terminal_cost_estimate = estimate_terminal_cost(current_belief, U, X, O, P, Z, C, 
                                                      gamma, rollout_horizon)
        total_cost += gamma^lookahead_horizon * terminal_cost_estimate
    end
    
    return total_cost
end

function estimate_terminal_cost(b::Vector{Float64}, U::Vector{Int}, X::Vector{Int}, O::Vector{Int},
                               P::Array{Float64,3}, Z::Matrix{Float64}, C::Matrix{Float64},
                               gamma::Float64, horizon::Int)::Float64
    """
    Estimates terminal cost using Monte Carlo simulation following base policy
    """
    
    current_belief = copy(b)
    total_cost = 0.0
    
    for t in 1:horizon
        u_t = base_policy()
        cost_t = POMDPUtil.expected_cost(current_belief, u_t, C, X)
        total_cost += gamma^(t-1) * cost_t
        
        if t < horizon
            o_t = POMDPUtil.sample_observation(current_belief, u_t, O, X, P, Z)
            current_belief = POMDPUtil.belief_operator(o_t, u_t, current_belief, X, Z, P)
        end
    end
    
    return total_cost
end

function base_policy()::Int
    """
    Deterministic base policy that always selects control=0
    """
    return 0
end