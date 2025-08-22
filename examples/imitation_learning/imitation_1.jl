using Pkg
Pkg.activate(".")
using JLD2, FileIO
using Flux
using Flux: train!
using Statistics
using Random

# Set random seed for reproducibility
Random.seed!(42)

# Print current working directory for debugging
println("Current working directory: $(pwd())")

# Load the training data
println("Loading training data...")
#data_file = "./data/multiagent_rollout_K10_T100_episodes250_eta0.2_pa0.05.jld2"
data_file = "./checkpoints/multiagent_rollout_K10_T100_episodes10000_eta0.2_pa0.05.jld2"

if !isfile(data_file)
    error("Data file not found: $data_file\nPlease run the data collection script first.")
end

@load data_file X Y beliefs controls K T num_episodes eta p_a alpha threshold

println("Loaded data:")
println("  Input shape (beliefs): $(size(X))")
println("  Output shape (controls): $(size(Y))")
println("  Number of samples: $(size(X, 2))")
println("  Number of components: $K")

# Convert to Float32 for Flux
X = Float32.(X)
Y = Float32.(Y)

# Split data into train/validation sets
n_samples = size(X, 2)
n_train = Int(floor(0.8 * n_samples))
indices = randperm(n_samples)
train_idx = indices[1:n_train]
val_idx = indices[n_train+1:end]

X_train = X[:, train_idx]
Y_train = Y[:, train_idx]
X_val = X[:, val_idx]
Y_val = Y[:, val_idx]

println("\nData split:")
println("  Training samples: $(size(X_train, 2))")
println("  Validation samples: $(size(X_val, 2))")

# Define the neural network model
# Multi-output binary classification (K independent sigmoid outputs)

model = Chain(
    Dense(K, 64, relu),
    Dense(64, 64, relu),
    Dropout(0.2),
    Dense(64, 64, relu),
    Dense(64, 32, relu),
    Dense(32, K, sigmoid)
)

#model = Chain(
    #Dense(K, 64, relu),
    #Dense(64, 128, relu),
    #Dense(128, 128, relu),
    #Dropout(0.2),
    #Dense(128, 128, relu),
    #Dense(128, 64, relu),
    #Dense(64, 32, relu),    
    #Dense(32, K, sigmoid)
#)

println("\nModel architecture:")
println(model)

# Define loss function (binary cross-entropy for each component)
function loss(x, y)
    y_pred = model(x)
    return Flux.Losses.binarycrossentropy(y_pred, y)
end

# Define accuracy metric
function accuracy(x, y)
    y_pred = model(x)
    y_pred_binary = y_pred .> 0.5
    return mean(y_pred_binary .== y)
end

# Setup optimizer
learning_rate = 0.0001
optimizer = Adam(learning_rate)
opt_state = Flux.setup(optimizer, model)

# Training parameters
epochs = 1000
batch_size = 2048
rolling_avg_window = 50

# Create data loaders
train_data = Flux.DataLoader((X_train, Y_train), batchsize=batch_size, shuffle=true)
val_data = (X_val, Y_val)

# Initialize rolling average tracking
train_loss_history = Float64[]
val_loss_history = Float64[]
train_acc_history = Float64[]
val_acc_history = Float64[]

println("\nStarting training...")
println("Epochs: $epochs, Batch size: $batch_size, Learning rate: $learning_rate, Optimizer: Adam")
println("Rolling average window: $rolling_avg_window")
println()

# Helper function to compute rolling average
function rolling_average(history, window_size)
    if length(history) == 0
        return 0.0
    end
    start_idx = max(1, length(history) - window_size + 1)
    return mean(history[start_idx:end])
end

# Training loop
best_val_loss = Inf
for epoch in 1:epochs
    # Training
    train_losses = Float64[]
    
    for (x_batch, y_batch) in train_data
        # Compute loss and gradients
        l, grads = Flux.withgradient(model) do m
            y_pred = m(x_batch)
            Flux.Losses.binarycrossentropy(y_pred, y_batch)
        end
        push!(train_losses, l)
        
        # Update parameters
        Flux.update!(opt_state, model, grads[1])
    end
    
    # Validation
    val_loss = loss(val_data...)
    train_acc = accuracy(X_train, Y_train)
    val_acc = accuracy(val_data...)
    
    # Update history for rolling averages
    avg_train_loss = mean(train_losses)
    push!(train_loss_history, avg_train_loss)
    push!(val_loss_history, val_loss)
    push!(train_acc_history, train_acc)
    push!(val_acc_history, val_acc)
    
    # Track best model
    if val_loss < best_val_loss
        global best_val_loss = val_loss
    end
    
    # Print progress with rolling averages
    #if epoch % 10 == 0 || epoch == 1
        #rolling_train_loss = rolling_average(train_loss_history, rolling_avg_window)
        #rolling_val_loss = rolling_average(val_loss_history, rolling_avg_window)
        #rolling_train_acc = rolling_average(train_acc_history, rolling_avg_window)
        #rolling_val_acc = rolling_average(val_acc_history, rolling_avg_window)
        
        #println("Epoch $epoch/$epochs:")
        #println("  Train Loss: $(round(rolling_train_loss, digits=4))")
        #println("  Val Loss:   $(round(rolling_val_loss, digits=4))")
        #println("  Train Acc:  $(round(rolling_train_acc, digits=4))")
        #println("  Val Acc:    $(round(rolling_val_acc, digits=4))")
        #println()
        #println("$epoch $(round(rolling_train_loss, digits=4)) $(round(rolling_val_loss, digits=4)) $(round(rolling_train_acc, digits=4)) $(round(rolling_val_acc, digits=4))")
    #end
    rolling_train_loss = rolling_average(train_loss_history, rolling_avg_window)
    rolling_val_loss = rolling_average(val_loss_history, rolling_avg_window)
    rolling_train_acc = rolling_average(train_acc_history, rolling_avg_window)
    rolling_val_acc = rolling_average(val_acc_history, rolling_avg_window)
    println("$epoch $(round(rolling_train_loss, digits=4)) $(round(rolling_val_loss, digits=4)) $(round(rolling_train_acc, digits=4)) $(round(rolling_val_acc, digits=4)) $(round(avg_train_loss, digits=4)) $(round(val_loss, digits=4)) $(round(train_acc, digits=4)) $(round(val_acc, digits=4))")
end

# Final evaluation
final_train_acc = accuracy(X_train, Y_train)
final_val_acc = accuracy(val_data...)
final_train_loss = loss(X_train, Y_train)
final_val_loss = loss(val_data...)

println("Training completed!")
println("\nFinal Results:")
println("  Train Loss: $(round(final_train_loss, digits=4))")
println("  Train Acc:  $(round(final_train_acc, digits=4))")
println("  Val Loss:   $(round(final_val_loss, digits=4))")
println("  Val Acc:    $(round(final_val_acc, digits=4))")

# Save the trained model
model_file = "./models/imitation_model_K$(K).jld2"
@save model_file model
println("\nModel saved to: $model_file")

# Test on a few sample predictions
#println("\nSample predictions:")
#n_samples_test = min(5, size(X_val, 2))
#for i in 1:n_samples_test
    #belief = X_val[:, i]
    #true_control = Y_val[:, i]
    #pred_control = model(belief)
    #pred_binary = pred_control .> 0.5
    
    #println("Sample $i:")
    #println("  Belief:     $(round.(belief, digits=3))")
    #println("  True:       $(Int.(true_control))")
    #println("  Predicted:  $(round.(pred_control, digits=3))")
    #println("  Binary:     $(Int.(pred_binary))")
    #println("  Match:      $(pred_binary == true_control)")
    #println()
#end
