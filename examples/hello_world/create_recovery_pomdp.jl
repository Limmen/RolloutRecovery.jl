using Pkg
Pkg.activate(".")
using RolloutRecovery

K = 5 
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