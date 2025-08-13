const PROB_THRESHOLD = 1e-10

function rollout_policy(b::Vector{Float64}, U::Vector{Int}, X::Vector{Int}, O::Vector{Int},
    P::Array{Float64,3}, Z::Matrix{Float64}, C::Matrix{Float64},
    alpha::Float64, lookahead_horizon::Int, rollout_horizon::Int,
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
        q_value = compute_q_value(b, u, U, X, O, P, Z, C,
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
    alpha::Float64, lookahead_horizon::Int, rollout_horizon::Int,
    num_simulations::Int)::Float64
    """
    Computes Q-value for control u in belief state b using lookahead optimization
    """

    immediate_cost = POMDPUtil.expected_cost(b, u, C, X)

    if lookahead_horizon == 0
        # No lookahead, just immediate cost plus rollout
        if rollout_horizon > 0
            rollout_cost = 0.0
            for sim in 1:num_simulations
                sim_cost = simulate_rollout(b, P, Z, C, X, O, alpha, rollout_horizon)
                rollout_cost += sim_cost
            end
            rollout_cost /= num_simulations
            return immediate_cost + alpha * rollout_cost + alpha^(rollout_horizon + 1) * terminal_cost(b, C, X)
        else
            return immediate_cost + alpha * terminal_cost(b, C, X)
        end
    end

    future_cost = 0.0

    @inbounds for o in O
        prob_o = POMDPUtil.compute_observation_probability(b, u, o, X, P, Z)

        if prob_o > PROB_THRESHOLD
            b_next = POMDPUtil.belief_operator(o, u, b, X, Z, P)
            
            # Recursively compute Q-value with reduced lookahead horizon
            min_expected_cost = Inf
            @inbounds for u_next in U
                q_next = compute_q_value(b_next, u_next, U, X, O, P, Z, C,
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
                         alpha::Float64, horizon::Int)::Float64
    """
    Simulates rollout following base policy for specified horizon
    """
    
    current_belief = copy(b)
    total_cost = 0.0
    alpha_power = 1.0
    
    @inbounds for t in 1:horizon
        u_t = base_policy()
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

function terminal_cost(b::Vector{Float64}, C::Matrix{Float64}, X::Vector{Int})::Float64
    """
    Computes terminal cost as expected cost at the belief state using base policy
    """
    u_terminal = base_policy()
    return POMDPUtil.expected_cost(b, u_terminal, C, X)
end

@inline function base_policy()::Int
    """
    Deterministic base policy that always selects control=0
    """
    return 0
end

function run_rollout_simulation(b_initial::Vector{Float64}, U::Vector{Int}, X::Vector{Int}, O::Vector{Int},
                               P::Array{Float64,3}, Z::Matrix{Float64}, C::Matrix{Float64},
                               alpha::Float64, lookahead_horizon::Int, rollout_horizon::Int,
                               num_simulations::Int, T::Int)::NamedTuple{(:controls, :observations, :beliefs, :costs, :total_cost), Tuple{Vector{Int}, Vector{Int}, Vector{Vector{Float64}}, Vector{Float64}, Float64}}
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
    - alpha: Discount factor
    - lookahead_horizon: Lookahead horizon ℓ
    - rollout_horizon: Rollout horizon m
    - num_simulations: Number of Monte Carlo simulations L
    - T: Number of time steps to simulate
    
    Returns:
    - Tuple of (controls, observations, beliefs, costs, total_cost)
    """
    
    controls = Vector{Int}(undef, T)
    observations = Vector{Int}(undef, T)
    beliefs = Vector{Vector{Float64}}(undef, T+1)
    costs = Vector{Float64}(undef, T)
    
    beliefs[1] = copy(b_initial)
    current_belief = copy(b_initial)
    total_cost = 0.0
    alpha_power = 1.0
    
    @inbounds for t in 1:T
        u_t = rollout_policy(current_belief, U, X, O, P, Z, C,
                            alpha, lookahead_horizon, rollout_horizon, num_simulations)
        controls[t] = u_t
        
        cost_t = POMDPUtil.expected_cost(current_belief, u_t, C, X)
        costs[t] = cost_t
        total_cost += alpha_power * cost_t
        alpha_power *= alpha
        
        o_t = POMDPUtil.sample_observation(current_belief, u_t, O, X, P, Z)
        observations[t] = o_t
        
        if t < T
            current_belief = POMDPUtil.belief_operator(o_t, u_t, current_belief, X, Z, P)
            beliefs[t+1] = copy(current_belief)
        end
    end
    
    return (controls=controls, observations=observations, beliefs=beliefs, 
            costs=costs, total_cost=total_cost)
end