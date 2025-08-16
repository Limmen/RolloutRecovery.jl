using Pkg
Pkg.activate(".")
using RolloutRecovery
using Statistics

K = 7
n = 1000
eta = 0.2
p_a = 0.05
p_c = 0.5

initial_state_vec = NTuple{K,Int}(zeros(Int, K))

A = RecoveryPOMDP.generate_erdos_renyi_graph(K, p_c)

println("Num states: $(2^K), num controls: $(2^K)")

alpha = 0.95
lookahead_horizon = 1
rollout_horizon = 5
num_simulations = 10
T = 1
num_lookahead_samples = 10
num_particles = 50
eval_samples = 1
threshold = 0.25

println("Running rollout simulation with T=$T time steps, eval_samples=$eval_samples...")
println("Parameters: alpha=$alpha, lookahead_horizon=$lookahead_horizon, rollout_horizon=$rollout_horizon, num_simulations=$num_simulations")

start_time = time()
average_cost = MonteCarloRollout.run_rollout_simulation(initial_state_vec, K, n, A, p_a, eta, 
                                                      alpha, lookahead_horizon, rollout_horizon, 
                                                      num_simulations, num_lookahead_samples, num_particles, T, eval_samples, threshold)
end_time = time()
execution_time = end_time - start_time

println("Execution time: $(round(execution_time, digits=4)) seconds")
println("Adjacency matrix A: $(A)")
println("Average total discounted cost: $(round(average_cost, digits=4))")

