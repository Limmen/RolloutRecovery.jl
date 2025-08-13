using Pkg
Pkg.activate(".")
using RolloutRecovery
using Statistics

K = 3
n = 5
eta = 2.0
p_a = 0.2
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

println("Num states: $(size(X, 1)), num controls: $(size(U, 1)), num observations: $(size(O, 1))")

alpha = 0.95
lookahead_horizon = 1
rollout_horizon = 5
num_simulations = 5
T = 1

println("Running rollout simulation with T=$T time steps...")
println("Parameters: alpha=$alpha, lookahead_horizon=$lookahead_horizon, rollout_horizon=$rollout_horizon, num_simulations=$num_simulations")

start_time = time()
result = RolloutRecovery.run_rollout_simulation(b0, U, X, O, P, Z, C, 
                                               alpha, lookahead_horizon, rollout_horizon, 
                                               num_simulations, T)
end_time = time()
execution_time = end_time - start_time

println("Simulation completed!")
println("Execution time: $(round(execution_time, digits=4)) seconds")
println("Total discounted cost: $(result.total_cost)")
println("Controls selected: $(result.controls)")
println("Average cost per step: $(mean(result.costs))")

