using Pkg
Pkg.activate(".")
using RolloutRecovery
using Statistics
using JLD2, FileIO
using CSV, DataFrames

K = 20
n = 1000
eta = 0.2
p_a = 0.05
p_c = 0.5

initial_state_vec = tuple(zeros(Int, K)...)  
A = RecoveryPOMDP.generate_erdos_renyi_graph(K, p_c)

println("K=$K components")

alpha = 0.95
lookahead_horizon = 0
rollout_horizon = 5
num_simulations = 10
T = 100
num_lookahead_samples = 10
num_particles = 50
num_episodes = 10000
threshold = 0.9
checkpoint_interval = 100 

println("Running data collection with T=$T time steps, num_episodes=$num_episodes...")
println("Parameters: alpha=$alpha, lookahead_horizon=$lookahead_horizon, rollout_horizon=$rollout_horizon, num_simulations=$num_simulations")
println("Checkpoint interval: every $checkpoint_interval episodes")

start_time = time()

# Create directories for data and checkpoints
data_dir = "data"
checkpoint_dir = "checkpoints"
graphs_dir = "graphs"
if !isdir(data_dir)
    mkdir(data_dir)
end
if !isdir(checkpoint_dir)
    mkdir(checkpoint_dir)
end
if !isdir(graphs_dir)
    mkdir(graphs_dir)
end

# Save adjacency matrix to graphs folder before starting data collection
graph_file = joinpath(graphs_dir, "$(K)_A.jld2")
@save graph_file A
println("Saved adjacency matrix to: $graph_file")

# Helper function to save checkpoint
function save_checkpoint(beliefs, controls, episode_num, filename_base)
    checkpoint_file = joinpath(checkpoint_dir, filename_base * ".jld2")
    
    # Convert to matrices for saving (use same variable names as final dataset)
    X = hcat(beliefs...)
    Y_vectors = [collect(controls[i]) for i in 1:length(controls)]
    Y = hcat(Y_vectors...)
    
    @save checkpoint_file X Y beliefs controls K T num_episodes eta p_a alpha threshold
    println("Saved checkpoint at episode $episode_num to: $checkpoint_file")
end

# Modified data collection with checkpointing
filename_base = "multiagent_rollout_K$(K)_T$(T)_episodes$(num_episodes)_eta$(eta)_pa$(p_a)"
beliefs = Vector{Vector{Float64}}()
controls = Vector{NTuple{K,Int}}()

# Collect data in chunks
episodes_completed = 0
while episodes_completed < num_episodes
    global episodes_completed  # Fix scoping issue
    
    # Calculate episodes for this chunk
    episodes_remaining = num_episodes - episodes_completed
    chunk_size = min(checkpoint_interval, episodes_remaining)
    
    println("Collecting episodes $(episodes_completed + 1) to $(episodes_completed + chunk_size)...")
    
    # Collect data for this chunk
    chunk_beliefs, chunk_controls = MonteCarloRolloutMultiagent.collect_data(initial_state_vec, K, n, A, p_a, 
                                                               alpha, lookahead_horizon, rollout_horizon, 
                                                               num_simulations, num_lookahead_samples, num_particles, T, chunk_size, threshold, eta)
    
    # Append to main collections
    append!(beliefs, chunk_beliefs)
    append!(controls, chunk_controls)
    
    episodes_completed += chunk_size
    
    # Save checkpoint
    if episodes_completed % checkpoint_interval == 0 || episodes_completed == num_episodes
        save_checkpoint(beliefs, controls, episodes_completed, filename_base)
    end
end
end_time = time()
execution_time = end_time - start_time

println("\nData collection completed!")
println("Collected $(length(beliefs)) trajectory points")
println("Each belief vector has length: $(length(beliefs[1]))")
println("Each control vector has length: $(length(controls[1]))")
println("Total execution time: $(round(execution_time, digits=2)) seconds")

# Display some sample data
println("\nSample trajectory data:")
for i in 1:min(5, length(beliefs))
    println("Step $i:")
    println("  Belief: $(round.(beliefs[i], digits=3))")
    println("  Control: $(controls[i])")
end


# Convert beliefs to matrix format (features x samples) for Flux
# Each column is a sample, each row is a feature (belief for each component)
X = hcat(beliefs...)  # K x N matrix where N = T * num_episodes
println("Input matrix X shape: $(size(X))")

# Convert controls to matrix format 
# Keep controls as binary vectors (K x N matrix) for multi-output regression/classification
Y_vectors = [collect(controls[i]) for i in 1:length(controls)]
Y = hcat(Y_vectors...)  # K x N matrix where each column is a control vector
println("Output matrix Y shape: $(size(Y))")
println("Each control component is binary (0 or 1), suitable for K independent binary classifiers")

# Save final complete dataset
jld2_file = joinpath(data_dir, filename_base * ".jld2")
@save jld2_file X Y beliefs controls K T num_episodes eta p_a alpha threshold
println("Saved JLD2 data to: $jld2_file")


df_data = DataFrame()
for k in 1:K
    df_data[!, "belief_$k"] = X[k, :]
end
for k in 1:K
    df_data[!, "control_$k"] = Y[k, :]
end
df_data[!, "episode"] = repeat(1:num_episodes, inner=T)
df_data[!, "timestep"] = repeat(1:T, outer=num_episodes)

csv_file = joinpath(data_dir, filename_base * ".csv")
CSV.write(csv_file, df_data)
println("Saved CSV data to: $csv_file")

metadata = Dict(
    "K" => K,
    "T" => T,
    "num_episodes" => num_episodes,
    "eta" => eta,
    "p_a" => p_a,
    "p_c" => p_c,
    "alpha" => alpha,
    "threshold" => threshold,
    "lookahead_horizon" => lookahead_horizon,
    "rollout_horizon" => rollout_horizon,
    "num_simulations" => num_simulations,
    "num_particles" => num_particles,
    "checkpoint_interval" => checkpoint_interval,
    "total_samples" => length(beliefs),
    "input_dim" => K,
    "output_dim" => K,
    "execution_time" => execution_time,
    "graph_file" => graph_file
)

metadata_file = joinpath(data_dir, filename_base * "_metadata.jld2")
@save metadata_file metadata
println("Saved metadata to: $metadata_file")

