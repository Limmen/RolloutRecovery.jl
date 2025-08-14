using Pkg
Pkg.activate(".")
using RolloutRecovery
using Statistics

K = 13
n = 1000
eta = 0.25
p_a = 0.1
p_c = 0.5

println("Creating POMDP")
X, x_to_vec, vec_to_x = RecoveryPOMDP.generate_state_space(K)
U, u_to_vec, vec_to_u, U_local = RecoveryPOMDP.generate_control_space(K)
x0 = RecoveryPOMDP.initial_state(K, vec_to_x)
b0 = RecoveryPOMDP.initial_belief(K, X, vec_to_x)
C = RecoveryPOMDP.generate_cost_matrix(X, U, x_to_vec, u_to_vec, eta)
A = RecoveryPOMDP.generate_erdos_renyi_graph(K, p_c)

println("Num states: $(size(X, 1)), num controls: $(size(U, 1))")

alpha = 0.95
lookahead_horizon = 1
rollout_horizon = 0
num_simulations = 0
T = 100
num_lookahead_samples = 10
num_particles = 50
eval_samples = 1
threshold = 0.25

println("Running rollout simulation with T=$T time steps, eval_samples=$eval_samples...")
println("Parameters: alpha=$alpha, lookahead_horizon=$lookahead_horizon, rollout_horizon=$rollout_horizon, num_simulations=$num_simulations")

start_time = time()
average_cost = MonteCarloRolloutMultiagentAsync.run_rollout_simulation(b0, U, X, K, n, x_to_vec, u_to_vec, vec_to_x, vec_to_u, A, p_a, C, 
                                                      alpha, lookahead_horizon, rollout_horizon, 
                                                      num_simulations, num_lookahead_samples, num_particles, T, eval_samples, threshold)
end_time = time()
execution_time = end_time - start_time

println("Execution time: $(round(execution_time, digits=4)) seconds")
println("Adjacency matrix A: $(A)")
println("Average total discounted cost: $(round(average_cost, digits=4))")

