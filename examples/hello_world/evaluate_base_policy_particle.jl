using Pkg
Pkg.activate(".")
using RolloutRecovery
using Distributions

K = 2
n = 10
eta = 0.25
p_a = 0.1
p_c = 0.1

X, x_to_vec, vec_to_x = RecoveryPOMDP.generate_state_space(K)
U, u_to_vec, vec_to_u, U_local = RecoveryPOMDP.generate_control_space(K)
O, o_to_vec, vec_to_o = RecoveryPOMDP.generate_observation_space(n, K)
x0 = RecoveryPOMDP.initial_state(K, vec_to_x)
b0 = RecoveryPOMDP.initial_belief(K, X, vec_to_x)
C = RecoveryPOMDP.generate_cost_matrix(X, U, x_to_vec, u_to_vec, eta)
A = RecoveryPOMDP.generate_erdos_renyi_graph(K, p_c)
P = RecoveryPOMDP.generate_transition_tensor(p_a, X, U, x_to_vec, u_to_vec, K, A)
Z = RecoveryPOMDP.generate_observation_tensor(n, X, K, x_to_vec, o_to_vec, O)
println(A)
println("Num states: $(size(X, 1)), num controls: $(size(U, 1)), num observations: $(size(O, 1))")

function evaluate_threshold_policy(horizon::Int, threshold::Float64, num_samples::Int,
    b_initial::Vector{Float64}, X::Vector{Int}, O::Vector{Int},
    P::Array{Float64,3}, Z::Matrix{Float64}, C::Matrix{Float64},
    x_to_vec::Dict{Int,NTuple{K,Int}}, u_to_vec::Dict{Int,NTuple{K,Int}},
    vec_to_u::Dict{NTuple{K,Int},Int}, alpha::Float64=0.95, M::Int=100)::Float64 where K
    
    intrusion_dist, no_intrusion_dist = RecoveryPOMDP.generate_component_observation_distributions(n)
    total_costs = Vector{Float64}(undef, num_samples)

    @inbounds for sample in 1:num_samples
        particles, weights = BootstrapParticleFilter.belief_to_particles(b_initial, X, M)
        true_state = X[rand(Categorical(b_initial))]

        total_cost = 0.0
        alpha_power = 1.0

        @inbounds for t in 1:horizon
            # Compute marginal belief for each component being compromised using particles
            control_vec = Vector{Int}(undef, K)
            @inbounds for k in 1:K
                marginal_belief_k = 0.0
                @inbounds for i in 1:M
                    state = particles[1, i]
                    state_vec = x_to_vec[state]
                    if state_vec[k] == 1  # Component k is compromised in this state
                        marginal_belief_k += weights[i]
                    end
                end
                control_vec[k] = marginal_belief_k > threshold ? 1 : 0
            end
            control_tuple = NTuple{K,Int}(control_vec)
            u_t = vec_to_u[control_tuple]

            cost_t = C[true_state+1, u_t+1]
            total_cost += alpha_power * cost_t
            alpha_power *= alpha

            if t < horizon
                true_state_vec = x_to_vec[true_state]
                o_t = RecoveryPOMDP.sample_observation_vector(true_state_vec, intrusion_dist, no_intrusion_dist)
                
                true_state = RecoveryPOMDP.sample_next_state_from_transition_probs(true_state, u_t, p_a, 
                                                                                  x_to_vec, u_to_vec, vec_to_x, A)
                
                particles, weights = BootstrapParticleFilter.particle_filter_update(particles, weights, 
                                                                           u_t, o_t, K, X, x_to_vec, u_to_vec, vec_to_x, A, p_a, intrusion_dist, no_intrusion_dist)
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
num_samples = 2000
alpha = 0.95
M = 50  # Number of particles

for threshold in 0.0:0.1:1.0
    local start_time = time()
    local avg_cost = evaluate_threshold_policy(horizon, threshold, num_samples,
        b0, X, O, P, Z, C, x_to_vec, u_to_vec, vec_to_u, alpha, M)
    local end_time = time()
    
    local execution_time = end_time - start_time
    println("Threshold: $threshold, Average cost: $(round(avg_cost, digits=4)), Time: $(round(execution_time, digits=2))s")
end