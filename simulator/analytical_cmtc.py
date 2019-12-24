import numpy as np
def solve_cmtc(Q):
	Qt = Q.transpose()
	s = Qt[0].size
	Qt[s-1] = 1
	e = np.zeros((s,1))
	e[s-1] = 1
	return np.linalg.solve(Qt,e)
 
# gera metricas analiticas para fila com duas classes de prioridade, sem interrupção, com taxas de entrada e serviço heterogêneas
def q8(l1,l2,m1,m2):
  d = {}
  d['l'] = l = l1+l1
  d['p1'] = p1 = l1 / l
  d['p2'] = p2 = 1 - p1
  d['r1'] = r1 = l1/m1
  d['r2'] = r2 = l2/m2
  d['r'] = r = r1 + r2
  d['x1'] = x1 = 1/m1
  d['x2'] = x2 = 1/m2
  d['x'] = x = (r1/r)*x1 + (r2/r)*x2
  d['xr1'] = xr1 = 1/m1
  d['xr2'] = xr2 = 1/m2
  d['xr'] = xr = (r1/r)*xr1 + (r2/r)*xr2
  d['w1'] = w1 = (r1*xr1 + r2*xr2)/(1 - r1)
  d['w2'] = w2 = (r1*xr1 + r2*xr2 + r1*w1)/(1 - r1 - r2)
  d['w'] = w = p1*w1 + p2*w2
  d['nq'] = nq = l*w
  d['u'] = u = r*xr/(1-r)
  d['g'] = g = x/(1-r)
  return d

import math

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

def mount_tax_natrix_Q_case2(lamb1, lamb2, mu1, mu2, n_layers):
  
  n_estates = 0
  for i in range(0,n_layers): 
    n_estates = n_estates + (i + 1)

  #taxes matrix Q
  Q = np.zeros((n_estates,n_estates))
  
  #fill matrix for 1st layer N=0
  Q[0][0] = -(lamb1 + lamb2)  
  Q[0][1] = lamb1
  Q[0][2] = lamb2

  
  #fill matrix for layers N=1..n_layers-1
  m = 1; #starts from line 1
  #for each layer
  for N in range(1,n_layers-1):
    #fill (N+1) estates for each layer
    for k in range(0,N+1):
      if k==0:
        Q[m][m-N] = mu1
        Q[m][m] = -(lamb1+lamb2+mu1)
        Q[m][m+(N+1)] = lamb1
        Q[m][m+(N+2)] = lamb2
      if k>0 and k<(N):
        Q[m][m-(N+1)] = mu2
        Q[m][m-N] = mu1
        Q[m][m] = -(lamb1+lamb2+mu1+mu2)
        Q[m][m+(N+1)] = lamb1
        Q[m][m+(N+2)] = lamb2  
      if k==(N):
        Q[m][m-(N+1)] = mu2
        Q[m][m] = -(lamb1+lamb2+mu2)
        Q[m][m+(N+1)] = lamb1
        Q[m][m+(N+2)] = lamb2
      m+=1
    
 
  
  #fill matrix for last layer N=n_layers
  N=n_layers - 1
  for k in range(0,N+1):
    if k==0:
      Q[m][m-N] = mu1
      Q[m][m] = -(mu1)
    if k>0 and k<(N):
      Q[m][m-(N+1)] = mu2
      Q[m][m-N] = mu1
      Q[m][m] = -(mu1+mu2)
    if k==(N):
      Q[m][m-(N+1)] = mu2
      Q[m][m] = -(mu2)
    m+=1 
  
  return Q 

def s(n):
  return int(n*(n+1)/2)

def idxtoan(i,j):
  base = s(i+j)
  return int(base + i)

def mtx3(l1,l2,m1,m2,nlayers):
  nstates = int(s(nlayers+1))
  Q = np.zeros((nstates,nstates))

  #case yellow
  Q[0][0] = -(l1+l2)
  Q[0][1] = l2
  Q[0][2] = l1

  for k in range(1,nlayers):
    #case red
    an = idxtoan(0,k)
    Q[an][an] = -(l1+l2+m2)
    Q[an][idxtoan(0,k+1)] = l2
    Q[an][idxtoan(1,k)] = l1
    Q[an][idxtoan(0,k-1)] = m2
    
    #case green
    for i in range(1,k+1):
      j = k-i
      an = idxtoan(i,j)
      Q[an][an] = -(l1+l2+m1)
      Q[an][idxtoan(i-1,j)] = m1
      Q[an][idxtoan(i+1,j)] = l1
      Q[an][idxtoan(i,j+1)] = l2

  #case dark gray
  an = idxtoan(0,nlayers)
  Q[an][an] = -m2
  Q[an][idxtoan(0,nlayers-1)] = m2

  #case light gray
  k=nlayers
  for i in range(1,nlayers+1):
    j = k-i
    an = idxtoan(i,j)
    Q[an][an] = -m1
    Q[an][idxtoan(i-1,j)] = m1
  
  return Q


def array_Nq_vs_lambda_case1_cmtc(mu, n_estates):

  lamb_array = np.array([0.05,0.1,0.15])#,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95])
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
    



def array_Nq_vs_lambda_case1_analytical(mu, n_estates,lamb_param=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]):

  lamb_array = np.array(lamb_param)
  s = lamb_array.size

  avg_Nq_analytical_array = np.empty(s)

  for k in range (0, s):
    #analytical solution
    rho = lamb_array[k]*(1/mu)
    avg_Nq_analytical_array[k] = lamb_array[k]*rho/(1-rho)
  
  return avg_Nq_analytical_array


def array_W_vs_lambda_case1_cmtc(mu, n_estates,lamb_param=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]):

  lamb_array = np.array(lamb_param)
  s = lamb_array.size

  avg_W_markov_array = np.empty(s)
  
  for k in range (0, s):
    #markov solution
    pi_matrix = solve_cmtc(mount_tax_natrix_Q_case1(lamb_array[k],mu, n_estates))
    avg_W_markov = 0
    for j in range (2,pi_matrix.size):
      avg_W_markov = avg_W_markov + (pi_matrix[j]*(j-1))/lamb_array[k]
    avg_W_markov_array[k] = avg_W_markov

  return avg_W_markov_array
  

def array_W_vs_lambda_case1_analytical(mu, n_estates,lamb_param):

  lamb_array = np.array(lamb_param)
  s = lamb_array.size

  avg_W_markov_array = np.empty(s)
  avg_W_analytical_array = np.empty(s)

  
  for k in range (0, s):
    #analytical solution
    rho = lamb_array[k]*(1/mu)
    avg_W_analytical_array[k] = rho/(1-rho)

  return avg_W_analytical_array

def array_Nq_vs_lambda_case2_cmtd(mu1, mu2, n_layers):

  lamb1_array = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5, 0.55, 0.59])
  lamb2 = 0.2

  s = lamb1_array.size
  avg_Nq_markov_array = np.empty(s)
  
  for k in range (0, s):
    
    # markov solution
    pi_matrix = solve_cmtc(mount_tax_natrix_Q_case2(lamb1_array[k],lamb2,mu1,mu2, n_layers))
    avg_Nq_markov = 0
    N = 2
    estate_index_in_layer_N = 0
    pi_sum = pi_matrix[0] + pi_matrix[1] + pi_matrix[2]
    for j in range (3,pi_matrix.size):
      avg_Nq_markov = avg_Nq_markov + pi_matrix[j]*(N-1)
      pi_sum = pi_sum + pi_matrix[j]
      estate_index_in_layer_N+=1
      if(estate_index_in_layer_N>N):
        N+=1
        estate_index_in_layer_N = 0
    avg_Nq_markov_array[k] = avg_Nq_markov
  
  return avg_Nq_markov_array







def array_Nq_vs_lambda_case2_analytical(mu1, mu2, n_layers):

  lamb1_array = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5, 0.55, 0.59])
  lamb2 = 0.2

  s = lamb1_array.size
  avg_Nq_analytical_array = np.empty(s)

  
  for k in range (0, s):
    # analytical solution
    p1 = lamb1_array[k]/(lamb1_array[k]+lamb2)
    p2 = lamb2/(lamb1_array[k]+lamb2)
    lamb = lamb1_array[k] + lamb2
    E_X = p1*(1/mu1) + p2*(1/mu2)  
    
    rho = lamb*E_X
    rho1 = lamb1_array[k]/mu1
    rho2 = lamb2/mu2
    
    E_Xr = (rho1/rho)*(1/mu1) + (rho2/rho)*(1/mu2)
    E_W1 = rho1*E_Xr/(1-rho)
    E_W2 = rho2*E_Xr/(1-rho)
    
    avg_Nq_analytical_array[k] = lamb*(p1*E_W1 + p2*E_W2)

  return avg_Nq_analytical_array

def array_W_vs_lambda_case2_cmtd(mu1, mu2, n_layers):

  lamb1_array = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.59])
  lamb2 = 0.2

  s = lamb1_array.size
  avg_W_markov_array = np.empty(s)
  
  for k in range (0, s): 
    #markov solution
    pi_matrix = solve_cmtc(mount_tax_natrix_Q_case2(lamb1_array[k],lamb2,mu1,mu2, n_layers))
    avg_W_markov = 0
    N = 2
    estate_index_in_layer_N = 0
    lamb = lamb1_array[k] + lamb2
    for j in range (3,pi_matrix.size):
      avg_W_markov = avg_W_markov + pi_matrix[j]*(N-1)/lamb
      estate_index_in_layer_N+=1
      if(estate_index_in_layer_N>N):
        N+=1
        estate_index_in_layer_N = 0
    avg_W_markov_array[k] = avg_W_markov
  return avg_W_markov_array

    

def array_W_vs_lambda_case2_analytical(mu1, mu2, n_layers):

  lamb1_array = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.59])
  lamb2 = 0.2

  s = lamb1_array.size
  avg_W_analytical_array = np.empty(s)
  
  for k in range (0, s):
    #analytical solution
    p1 = lamb1_array[k]/(lamb1_array[k]+lamb2)
    p2 = lamb2/(lamb1_array[k]+lamb2)
    lamb = lamb1_array[k] + lamb2
    E_X = p1*(1/mu1) + p2*(1/mu2)

    rho = lamb*E_X
    rho1 = lamb1_array[k]/mu1
    rho2 = lamb2/mu2
    
    E_Xr = (rho1/rho)*(1/mu1) + (rho2/rho)*(1/mu2)
    E_W1 = rho1*(1/mu1)/(1-rho)
    E_W2 = rho2*(1/mu2)/(1-rho)

    avg_W_analytical_array[k] = p1*E_W1 + p2*E_W2

  return avg_W_analytical_array


def cmtc_solution_nq_case2_priority(l1,l2,m1,m2,nlayers):
  Q = mtx3(l1,l2,m1,m2,nlayers)
  sol = solve_cmtc(Q)
  
  nq = 0
  for i in range(2,nlayers+1):
    nq+= (i-1)*sum(sol[idxtoan(0,i):(idxtoan(i,0)+1)])
  return nq

def cmtc_solution_W_case2_priority(l1,l2,m1,m2,nlayers):
  Q = mtx3(l1,l2,m1,m2,nlayers)
  sol = solve_cmtc(Q)
  
  w1,w2 = (0,0)
  for l in range(2,nlayers+1):
    for i in range(0,l+1):
      j = l-i
      sttpi = sol[idxtoan(i,j)]
      w1+=sttpi*i
      w2+=sttpi*j

  p1 = l1/(l1+l2)
  p2 = 1 - p1

  return p1*w1 + p2*w2

def array_Nq_vs_lambda_case2_priority_cmtd(nlayers):
  l1s = np.arange(.05, .6, .05)
  l2 = .2
  m1 = 1.0
  m2 = .5
  anal_sol = []
  cmtc_sol = []
  for l1 in l1s:
    cmtc_sol.append(cmtc_solution_W_case2_priority(l2,l1,m1,m2,nlayers))
  
  return cmtc_sol
  

def array_Nq_vs_lambda_case2_priority_analytical(nlayers):
  l1s = np.arange(.05, .6, .05)
  l2 = .2
  m1 = 1.0
  m2 = .5
  anal_sol = []
  cmtc_sol = []
  for l1 in l1s:
    anal_sol.append(q8(l1,l2,m1,m2)['nq'])
    #anal_sol.append(anal_solution_case2_priority(l1,l2,m1,m2))
  return anal_sol


def array_W_vs_lambda_case2_priority_cmtc(nlayers):
  l1s = np.arange(.05, .6, .05)
  l2 = .2
  m1 = 1.0
  m2 = .5
  anal_sol = []
  cmtc_sol = []
  for l1 in l1s:
    cmtc_sol.append(cmtc_solution_W_case2_priority(l2,l1,m1,m2,nlayers))
  
  return cmtc_sol

def array_W_vs_lambda_case2_priority_analytical(nlayers):
  l1s = np.arange(.05, .6, .05)
  l2 = .2
  m1 = 1.0
  m2 = .5
  anal_sol = []
  cmtc_sol = []
  for l1 in l1s:
    anal_sol.append(q8(l1,l2,m1,m2)['w'])
    #anal_sol.append((anal_solution_case2_priority(l1,l2,m1,m2) / (l1+l2)))
  return anal_sol  



