using Test

@testset "Recovery POMDP Tests" begin
    
    @testset "State Space Generation" begin
        K = 2
        X, x_to_vec, vec_to_x = RecoveryPOMDP.generate_state_space(K)
        
        @test length(X) == 2^K
        @test X == [0, 1, 2, 3]
        @test length(x_to_vec) == 2^K
        @test length(vec_to_x) == 2^K
        
        @test x_to_vec[0] == (0, 0)
        @test x_to_vec[1] == (1, 0)
        @test x_to_vec[2] == (0, 1)
        @test x_to_vec[3] == (1, 1)
        
        @test vec_to_x[(0, 0)] == 0
        @test vec_to_x[(1, 1)] == 3
    end
    
    @testset "Control Space Generation" begin
        K = 2
        U, u_to_vec, vec_to_u, action_space = RecoveryPOMDP.generate_control_space(K)
        
        @test length(U) == 2^K
        @test U == [0, 1, 2, 3]
        @test length(u_to_vec) == 2^K
        @test length(vec_to_u) == 2^K
        @test length(action_space) == K
        
        @test u_to_vec[0] == (0, 0)
        @test u_to_vec[3] == (1, 1)
        @test vec_to_u[(0, 1)] == 2
        @test action_space[1] == [0, 1]
    end
    
    @testset "Observation Space Generation" begin
        n, K = 2, 2
        O, o_to_vec, vec_to_o = RecoveryPOMDP.generate_observation_space(n, K)
        
        @test length(O) == (n + 1)^K
        @test O == [0, 1, 2, 3, 4, 5, 6, 7, 8]
        @test length(o_to_vec) == (n + 1)^K
        @test length(vec_to_o) == (n + 1)^K
        
        @test o_to_vec[0] == (0, 0)
        @test o_to_vec[8] == (2, 2)
        @test vec_to_o[(1, 0)] == 1
    end
    
    @testset "Initial State and Belief" begin
        K = 3
        X, x_to_vec, vec_to_x = RecoveryPOMDP.generate_state_space(K)
        
        x0 = RecoveryPOMDP.initial_state(K, vec_to_x)
        @test x0 == 0
        @test x_to_vec[x0] == (0, 0, 0)
        
        b0 = RecoveryPOMDP.initial_belief(K, X, vec_to_x)
        @test length(b0) == length(X)
        @test sum(b0) ≈ 1.0
        @test b0[x0 + 1] == 1.0
        @test all(b0[i] == 0.0 for i in 1:length(b0) if i != x0 + 1)
    end
    
    @testset "Erdős-Rényi Graph Generation" begin
        K = 4
        p_c = 0.5
        A = RecoveryPOMDP.generate_erdos_renyi_graph(K, p_c)
        
        @test size(A) == (K, K)
        @test all(A[i, i] == 0 for i in 1:K)
        @test A == A'
        @test all(x -> x in [0, 1], A)
    end
    
    @testset "Cost Function" begin
        K = 2
        X, x_to_vec, vec_to_x = RecoveryPOMDP.generate_state_space(K)
        U, u_to_vec, vec_to_u, _ = RecoveryPOMDP.generate_control_space(K)
        eta = 1.5
        
        cost_00 = RecoveryPOMDP.cost_function(0, 0, x_to_vec, u_to_vec, eta)
        @test cost_00 >= 0.0
        
        cost_33 = RecoveryPOMDP.cost_function(3, 3, x_to_vec, u_to_vec, eta)
        @test cost_33 >= 0.0
        
        C = RecoveryPOMDP.generate_cost_matrix(X, U, x_to_vec, u_to_vec, eta)
        @test size(C) == (length(X), length(U))
        @test all(C .>= 0.0)
    end
    
    @testset "Local Transition Probability" begin
        p_a = 0.1
        
        @test RecoveryPOMDP.local_transition_probability(0, 0, 1, p_a, 0) == 1.0
        @test RecoveryPOMDP.local_transition_probability(1, 1, 0, p_a, 0) == 1.0
        @test RecoveryPOMDP.local_transition_probability(1, 0, 0, p_a, 0) ≈ p_a
        @test RecoveryPOMDP.local_transition_probability(0, 0, 0, p_a, 0) ≈ 1.0 - p_a
        @test RecoveryPOMDP.local_transition_probability(0, 1, 1, p_a, 0) == 1.0
    end
    
    @testset "Transition Probability" begin
        K = 2
        X, x_to_vec, vec_to_x = RecoveryPOMDP.generate_state_space(K)
        U, u_to_vec, vec_to_u, _ = RecoveryPOMDP.generate_control_space(K)
        A = zeros(Int, K, K)
        p_a = 0.1
        
        prob = RecoveryPOMDP.transition_probability(0, 0, 0, K, p_a, x_to_vec, u_to_vec, A)
        @test prob >= 0.0
        @test prob <= 1.0
        
        prob_reset = RecoveryPOMDP.transition_probability(0, 3, 3, K, p_a, x_to_vec, u_to_vec, A)
        @test prob_reset == 1.0
    end
    
    @testset "Transition Tensor" begin
        K = 2
        X, x_to_vec, vec_to_x = RecoveryPOMDP.generate_state_space(K)
        U, u_to_vec, vec_to_u, _ = RecoveryPOMDP.generate_control_space(K)
        A = zeros(Int, K, K)
        p_a = 0.1
        
        P = RecoveryPOMDP.generate_transition_tensor(p_a, X, U, x_to_vec, u_to_vec, K, A)
        @test size(P) == (length(U), length(X), length(X))
        @test all(P .>= 0.0)
        @test all(P .<= 1.0)
        
        for u_idx in 1:length(U), x_idx in 1:length(X)
            row_sum = sum(P[u_idx, x_idx, :])
            @test isapprox(row_sum, 1.0, rtol=0.01)
        end
    end
    
    @testset "Observation Tensor" begin
        n, K = 1, 2
        X, x_to_vec, vec_to_x = RecoveryPOMDP.generate_state_space(K)
        O, o_to_vec, vec_to_o = RecoveryPOMDP.generate_observation_space(n, K)
        
        Z = RecoveryPOMDP.generate_observation_tensor(n, X, K, x_to_vec, o_to_vec, O)
        @test size(Z) == (length(X), length(O))
        @test all(Z .>= 0.0)
        @test all(Z .<= 1.0)
        
        for x_idx in 1:length(X)
            row_sum = sum(Z[x_idx, :])
            @test isapprox(row_sum, 1.0, rtol=0.01)
        end
    end
end