using Pkg
Pkg.activate(".")
using JLD2, FileIO
using Flux
using Statistics
using Random

# Set random seed for reproducibility
Random.seed!(42)

println("Current working directory: $(pwd())")

# Load the trained model
println("Loading trained model...")
model_file = "./models/imitation_model_K4.jld2"

if !isfile(model_file)
    error("Model file not found: $model_file\nPlease train the model first using imitation_1.jl")
end

@load model_file model

println("Model loaded successfully!")
println("Model architecture:")
println(model)

# Load test data (or create synthetic test data)
println("\nLoading test data...")
data_file = "./data/multiagent_rollout_K4_T100_episodes100_eta0.2_pa0.05.jld2"

if !isfile(data_file)
    error("Data file not found: $data_file\nPlease run the data collection script first.")
end

@load data_file X Y beliefs controls K T num_episodes eta p_a alpha threshold

# Convert to Float32 for Flux
X = Float32.(X)
Y = Float32.(Y)

println("Loaded test data:")
println("  Input shape (beliefs): $(size(X))")
println("  Output shape (controls): $(size(Y))")
println("  Number of samples: $(size(X, 2))")

# Function to make predictions
function predict_control(model, belief_vector)
    """
    Make a control prediction given a belief vector
    
    Args:
    - model: Trained Flux model
    - belief_vector: Vector of belief probabilities (length K)
    
    Returns:
    - predicted_probs: Predicted probabilities for each component (sigmoid outputs)
    - predicted_binary: Binary control decisions (0 or 1 for each component)
    """
    belief_vector = Float32.(belief_vector)
    predicted_probs = model(belief_vector)
    predicted_binary = predicted_probs .> 0.5
    return predicted_probs, predicted_binary
end

# Function to evaluate model performance
function evaluate_model(model, X_test, Y_test)
    """
    Evaluate model performance on test data
    """
    n_samples = size(X_test, 2)
    correct_predictions = 0
    total_elements = 0
    
    # Component-wise accuracy
    component_accuracy = zeros(K)
    
    for i in 1:n_samples
        belief = X_test[:, i]
        true_control = Y_test[:, i]
        
        pred_probs, pred_binary = predict_control(model, belief)
        
        # Overall accuracy
        correct_predictions += sum(pred_binary .== true_control)
        total_elements += K
        
        # Component-wise accuracy
        for k in 1:K
            if pred_binary[k] == true_control[k]
                component_accuracy[k] += 1
            end
        end
    end
    
    overall_accuracy = correct_predictions / total_elements
    component_accuracy ./= n_samples
    
    return overall_accuracy, component_accuracy
end

# Evaluate on full dataset
println("\nEvaluating model performance...")
overall_acc, comp_acc = evaluate_model(model, X, Y)

println("Performance metrics:")
println("  Overall accuracy: $(round(overall_acc, digits=4))")
println("  Component-wise accuracy:")
for k in 1:K
    println("    Component $k: $(round(comp_acc[k], digits=4))")
end

# Test on individual samples
println("\nSample predictions:")
n_test_samples = min(10, size(X, 2))
test_indices = rand(1:size(X, 2), n_test_samples)

for (i, idx) in enumerate(test_indices)
    belief = X[:, idx]
    true_control = Y[:, idx]
    
    pred_probs, pred_binary = predict_control(model, belief)
    
    println("Sample $i (index $idx):")
    println("  Belief:           $(round.(belief, digits=3))")
    println("  True control:     $(Int.(true_control))")
    println("  Predicted probs:  $(round.(pred_probs, digits=3))")
    println("  Predicted binary: $(Int.(pred_binary))")
    println("  Exact match:      $(pred_binary == true_control)")
    println("  Accuracy:         $(round(mean(pred_binary .== true_control), digits=3))")
    println()
end

# Interactive prediction function
function interactive_prediction()
    """
    Allow user to input custom belief vectors for prediction
    """
    println("\n" * "="^50)
    println("Interactive Prediction Mode")
    println("="^50)
    println("Enter belief probabilities for each component (0.0 to 1.0)")
    println("Press Ctrl+C to exit")
    
    while true
        try
            print("\nEnter belief vector (comma-separated, e.g., 0.1,0.8,0.3,0.2): ")
            input = readline()
            
            # Parse input
            belief_str = split(strip(input), ",")
            if length(belief_str) != K
                println("Error: Please enter exactly $K values")
                continue
            end
            
            belief = [parse(Float64, strip(s)) for s in belief_str]
            
            # Validate input
            if any(x -> x < 0 || x > 1, belief)
                println("Error: All values must be between 0.0 and 1.0")
                continue
            end
            
            # Make prediction
            pred_probs, pred_binary = predict_control(model, belief)
            
            println("Results:")
            println("  Input belief:     $(round.(belief, digits=3))")
            println("  Predicted probs:  $(round.(pred_probs, digits=3))")
            println("  Predicted action: $(Int.(pred_binary))")
            
            # Interpretation
            actions_taken = sum(pred_binary)
            println("  Interpretation:   $actions_taken out of $K components should be recovered")
            
        catch ex
            if isa(ex, InterruptException)
                println("\nExiting interactive mode...")
                break
            else
                println("Error parsing input: Please enter values in format: 0.1,0.8,0.3,0.2")
            end
        end
    end
end

# Offer interactive mode
println("\nWould you like to try interactive prediction mode? (y/n)")
print("Enter choice: ")
try
    choice = readline()
    if lowercase(strip(choice)) in ["y", "yes"]
        interactive_prediction()
    end
catch InterruptException
    println("Skipping interactive mode...")
end

println("\nInference completed!")
