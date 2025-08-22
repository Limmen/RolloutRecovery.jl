using Pkg
Pkg.activate(".")
using RolloutRecovery
using Distributions

K = 4
n = 10
eta = 0.25
p_a = 0.05
p_c = 0.5

X, x_to_vec, vec_to_x = RecoveryPOMDP.generate_state_space(K)
U, u_to_vec, vec_to_u, U_local = RecoveryPOMDP.generate_control_space(K)
O, o_to_vec, vec_to_o = RecoveryPOMDP.generate_observation_space(n, K)
x0 = RecoveryPOMDP.initial_state(K, vec_to_x)
b0 = RecoveryPOMDP.initial_belief(K, X, vec_to_x)
C = RecoveryPOMDP.generate_cost_matrix(X, U, x_to_vec, u_to_vec, eta)
A = RecoveryPOMDP.generate_erdos_renyi_graph(K, p_c)
P = RecoveryPOMDP.generate_transition_tensor(p_a, X, U, x_to_vec, u_to_vec, K, A)
Z = RecoveryPOMDP.generate_observation_tensor(n, X, K, x_to_vec, o_to_vec, O)

println("Num states: $(size(X, 1)), num controls: $(size(U, 1)), num observations: $(size(O, 1))")

function evaluate_threshold_policy(horizon::Int, threshold::Float64, num_samples::Int,
    b_initial::Vector{Float64}, X::Vector{Int}, O::Vector{Int},
    P::Array{Float64,3}, Z::Matrix{Float64}, C::Matrix{Float64},
    x_to_vec::Dict{Int,NTuple{K,Int}}, u_to_vec::Dict{Int,NTuple{K,Int}},
    vec_to_u::Dict{NTuple{K,Int},Int}, alpha::Float64=0.95)::Float64 where K
    total_costs = Vector{Float64}(undef, num_samples)

    @inbounds for sample in 1:num_samples
        current_belief = copy(b_initial)
        true_state = X[rand(Categorical(b_initial))]

        total_cost = 0.0
        alpha_power = 1.0

        @inbounds for t in 1:horizon
            # Compute marginal belief for each component being compromised
            control_vec = Vector{Int}(undef, K)
            @inbounds for k in 1:K
                marginal_belief_k = 0.0
                @inbounds for (state_idx, state) in enumerate(X)
                    state_vec = x_to_vec[state]
                    if state_vec[k] == 1  # Component k is compromised in this state
                        marginal_belief_k += current_belief[state_idx]
                    end
                end
                control_vec[k] = marginal_belief_k > threshold ? 1 : 0
            end
            control_tuple = NTuple{K,Int}(control_vec)
            u_t = vec_to_u[control_tuple]
            #println("Time step $t: Control = $control_vec, marginal beliefs = [$(join([round(sum(current_belief[i] for (i, s) in enumerate(X) if x_to_vec[s][k] == 1), digits=3) for k in 1:K], ", "))]")

            cost_t = C[true_state+1, u_t+1]
            total_cost += alpha_power * cost_t
            alpha_power *= alpha

            if t < horizon
                state_idx = true_state + 1
                observation_probs = Z[state_idx, :]
                o_t = O[rand(Categorical(observation_probs))]

                control_idx = u_t + 1
                current_state_idx = true_state + 1
                transition_probs = P[control_idx, current_state_idx, :]
                true_state = X[rand(Categorical(transition_probs))]

                current_belief = POMDPUtil.belief_operator(o_t, u_t, current_belief, X, Z, P)
            end
        end

        total_costs[sample] = total_cost
        
        if sample % 10 == 0 || sample == num_samples
            current_avg = sum(total_costs[1:sample]) / sample
            #println("Sample $sample/$num_samples, Current average cost: $(round(current_avg, digits=4))")
        end
    end

    return sum(total_costs) / num_samples
end

horizon = 100
num_samples = 1000
alpha = 0.95

for threshold in 0.0:0.1:1.0
    local start_time = time()
    local avg_cost = evaluate_threshold_policy(horizon, threshold, num_samples,
        b0, X, O, P, Z, C, x_to_vec, u_to_vec, vec_to_u, alpha)
    local end_time = time()
    
    local execution_time = end_time - start_time
    println("Threshold: $threshold, Average cost: $(round(avg_cost, digits=4)), Time: $(round(execution_time, digits=2))s")
end