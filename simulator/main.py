import plotGraf as plot
import analytical_cmtc as anal_mark
import sim_run as simulator
import numpy as np



n_estates = 100
mu1 = 1
#lamb_param_1=[0.05, 0.1, 0.15]#,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
lamb_param_1=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
caso_1_anal_Nq = anal_mark.array_Nq_vs_lambda_case1_analytical(mu1, n_estates,lamb_param_1)
caso_1_cmtc_Nqs = anal_mark.array_Nq_vs_lambda_case1_cmtc(mu1, n_estates)
caso_1_cmtc_N1qs = anal_mark.array_Nq_vs_lambda_case1_cmtc(mu1, n_estates)
caso_1_cmtc_Nq =[2.63157895e-03, 1.11111111e-01, 1.76470588e-01, 2.50000000e-01,        3.33333333e-01, 4.28571429e-01, 5.38461538e-01, 6.66666667e-01,        8.18181818e-01, 1.00000000e+00, 1.22222222e+00, 1.50000000e+00,        1.85714286e+00, 2.33333333e+00, 3.00000000e+00, 3.99999997e+00,        5.66665639e+00, 8.99705161e+00]
print('caso_1_anal_Nq')
print(caso_1_anal_Nq)
print('caso_1_cmtc_Nq')
print(caso_1_cmtc_Nq)

caso_1_simulator = simulator.agregate_methods_case1(lamb_param_1)
arr_w_interval = [res['w'].interval for res in caso_1_simulator]
arr_w_means = [res['w'].mean for res in caso_1_simulator]
arr_Nq_interval = [res['nq'].interval for res in caso_1_simulator]
arr_Nq_means = [res['nq'].mean for res in caso_1_simulator]

print('markov')
print(caso_1_cmtc_Nq)
print('caso_1_anal_Nq')
print(caso_1_anal_Nq)
plot.plot_E_vs_lambda('Nq', lamb_param_1, caso_1_anal_Nq,caso_1_cmtc_Nq , arr_Nq_means, arr_Nq_interval , 10,10, 'caso_1_Nq')


caso_1_analytic_w = anal_mark.array_W_vs_lambda_case1_analytical(mu1, n_estates,lamb_param_1)
caso_1_cmtc_w = anal_mark.array_W_vs_lambda_case1_cmtc(mu1, n_estates,lamb_param_1)

plot.plot_E_vs_lambda('W', lamb_param_1, caso_1_analytic_w, caso_1_cmtc_w, arr_w_means, arr_w_interval,  10,10, 'caso_1_W')
