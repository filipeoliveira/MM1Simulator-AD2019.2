import numpy as np

def solve_cmtc(Q):
	Qt = Q.transpose()
	s = Qt[0].size
	Qt[s-1] = 1
	e = np.zeros((s,1))
	e[s-1] = 1
	return np.linalg.solve(Qt,e)
 
def mount_tax_natrix_Q_case1(lamb, mu, n_estates):
  #taxes matrix Q
  Q = np.zeros((n_estates,n_estates))
  
  #fill matrix top line "1"
  Q[0][0] = -lamb  
  Q[0][1] = lamb

  #fill matrix botton line "N"
  Q[n_estates-1][n_estates-1] = -mu
  Q[n_estates-1][n_estates-2] = mu
  
  #fill matrix lines 2..N-1
  for i in range(1,n_estates-1):
    for j in range(i-1,i+2):
      if j < i:
        Q[i][j] = mu
      if i == j:
        Q[i][j] = -(lamb+mu)  
      if j > i:
        Q[i][j] = lamb

  return Q

def array_Nq_vs_lambda_case1_cmtc(mu, n_estates):

  lamb_array = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95])
  s = lamb_array.size

  avg_Nq_markov_array = np.empty(s)

  
  for k in range (0, s):
    #markov solution
    pi_matrix = solve_cmtc(mount_tax_natrix_Q_case1(lamb_array[k],mu, n_estates))
    avg_Nq_markov = 0
    for j in range (2,pi_matrix.size):
      avg_Nq_markov = avg_Nq_markov + pi_matrix[j]*(j-1)
    avg_Nq_markov_array[k] = avg_Nq_markov
    
    return avg_Nq_markov_array

print(array_Nq_vs_lambda_case1_cmtc(1, 100))