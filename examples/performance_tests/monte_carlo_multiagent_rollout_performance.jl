using Pkg
Pkg.activate(".")
using RolloutRecovery
using Statistics
using JLD2, FileIO

K = 40
n = 1000
eta = 0.2
p_a = 0.05
p_c = 0.5

initial_state_vec = tuple(zeros(Int, K)...)  
A = RecoveryPOMDP.generate_erdos_renyi_graph(K, p_c)

# Load adjacency matrix from graphs folder
#graph_file = "./graphs/$(K)_A.jld2"
#if !isfile(graph_file)
    #error("Adjacency matrix file not found: $graph_file\nPlease run the data collection script first to generate the graph.")
#end

#println("Loading adjacency matrix from: $graph_file")
#@load graph_file A

println("K=$K components")

alpha = 0.95
lookahead_horizon = 0
rollout_horizon = 5
num_simulations = 10
T = 100
num_lookahead_samples = 10
num_particles = 50
eval_samples = 100
threshold = 0.9

println("Running rollout simulation with T=$T time steps, eval_samples=$eval_samples...")
println("Parameters: alpha=$alpha, lookahead_horizon=$lookahead_horizon, rollout_horizon=$rollout_horizon, num_simulations=$num_simulations")

start_time = time()
average_cost = MonteCarloRolloutMultiagent.run_rollout_simulation(initial_state_vec, K, n, A, p_a, 
                                                      alpha, lookahead_horizon, rollout_horizon, 
                                                      num_simulations, num_lookahead_samples, num_particles, T, eval_samples, threshold, eta)
end_time = time()
execution_time = end_time - start_time

println("Execution time: $(round(execution_time, digits=4)) seconds")
println("Adjacency matrix A: $(A)")
println("Average total discounted cost: $(round(average_cost, digits=4))")

