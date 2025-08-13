using Test

@testset "POMDP Utility Functions Tests" begin
    
    @testset "Expected Cost Function" begin        
        X = [0, 1]
        U = [0, 1]
        C = [1.0 2.0; 
             3.0 4.0]
        b = [0.5, 0.5]

        expected_cost_0 = POMDPUtil.expected_cost(b, 0, C, X)
        @test expected_cost_0 ≈ 0.5 * 1.0 + 0.5 * 3.0  # 2.0
        
        expected_cost_1 = POMDPUtil.expected_cost(b, 1, C, X)
        @test expected_cost_1 ≈ 0.5 * 2.0 + 0.5 * 4.0  # 3.0
        
        b_certain = [1.0, 0.0]
        cost_certain = POMDPUtil.expected_cost(b_certain, 0, C, X)
        @test cost_certain ≈ 1.0
    end
    
    @testset "Bayes Filter Function" begin
        X = [0, 1]
    
        P = zeros(2, 2, 2)
        P[1, 1, 1] = 0.8
        P[1, 1, 2] = 0.2
        P[1, 2, 1] = 0.3
        P[1, 2, 2] = 0.7
                
        Z = [0.9 0.1;
             0.2 0.8]
        
        b = [0.6, 0.4]
                
        result = POMDPUtil.bayes_filter(0, 0, 0, b, X, Z, P)
        @test result >= 0.0
        @test result <= 1.0
        @test typeof(result) == Float64
    end
    
    @testset "Belief Operator Function" begin        
        X = [0, 1]
        
        P = zeros(2, 2, 2)
        P[1, 1, 1] = 0.7
        P[1, 1, 2] = 0.3
        P[1, 2, 1] = 0.4
        P[1, 2, 2] = 0.6
        
        Z = [0.8 0.2;
             0.3 0.7]
        
        b = [0.5, 0.5]
        
        b_prime = POMDPUtil.belief_operator(0, 0, b, X, Z, P)

        @test length(b_prime) == length(X)
        @test all(b_prime .>= 0.0)
        @test sum(b_prime) ≈ 1.0
        @test typeof(b_prime) == Vector{Float64}
    end
    
    @testset "Compute Observation Probability Function" begin
        X = [0, 1]
        
        P = zeros(2, 2, 2)
        P[1, 1, 1] = 0.7
        P[1, 1, 2] = 0.3
        P[1, 2, 1] = 0.4
        P[1, 2, 2] = 0.6
        
        Z = [0.8 0.2;
             0.3 0.7]
        
        b = [0.5, 0.5]
                
        prob_o0 = POMDPUtil.compute_observation_probability(b, 0, 0, X, P, Z)
        prob_o1 = POMDPUtil.compute_observation_probability(b, 0, 1, X, P, Z)
        
        @test prob_o0 >= 0.0
        @test prob_o1 >= 0.0
        @test prob_o0 + prob_o1 ≈ 1.0
        @test typeof(prob_o0) == Float64
        @test typeof(prob_o1) == Float64
                
        b_certain = [1.0, 0.0]
        prob_certain = POMDPUtil.compute_observation_probability(b_certain, 0, 0, X, P, Z)
        @test prob_certain >= 0.0
        @test prob_certain <= 1.0
    end
    
    @testset "Sample Observation Function" begin
        X = [0, 1]
        O = [0, 1]
        
        P = zeros(2, 2, 2)
        P[1, 1, 1] = 0.7
        P[1, 1, 2] = 0.3
        P[1, 2, 1] = 0.4
        P[1, 2, 2] = 0.6
        
        Z = [0.8 0.2;
             0.3 0.7]
        
        b = [0.5, 0.5]
            
        for _ in 1:10
            obs = POMDPUtil.sample_observation(b, 0, O, X, P, Z)
            @test obs in O
            @test typeof(obs) == Int
        end
            
        Z_zero = [0.0 0.0;
                  0.0 0.0]
        obs_zero = POMDPUtil.sample_observation(b, 0, O, X, P, Z_zero)
        @test obs_zero == O[1]  
        
        Z_det = [1.0 0.0;
                 1.0 0.0]
        obs_det = POMDPUtil.sample_observation(b, 0, O, X, P, Z_det)
        @test obs_det == 0
    end
    
    @testset "Edge Cases" begin        
        X_single = [0]
        b_single = [1.0]
        C_single = reshape([5.0], 1, 1)

        cost_single = POMDPUtil.expected_cost(b_single, 0, C_single, X_single)
        @test cost_single ≈ 5.0
        
        X = [0, 1]
        b_zero = [0.0, 0.0]
        C = [1.0 2.0; 3.0 4.0]

        cost_zero = POMDPUtil.expected_cost(b_zero, 0, C, X)
        @test cost_zero ≈ 0.0
    end
end