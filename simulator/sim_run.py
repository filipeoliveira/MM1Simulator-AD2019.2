#CASE 1
import simulator as s
import plot as plot

#lamb_array_case1 = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95])


#lamb1_array_case2 = np.array([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5, 0.55, 0.59])

lamb2_case2 = 0.2

mu1_case2 = 1

mu2_case2 = 0.5

def agregate_methods_case1(lamb_array):
  mu_case1 = 1
  results = []
  for lamb in lamb_array:
    results.append(s.run_simulation (lamb/2, lamb/2, mu_case1, mu_case1, 'exp'))
    print(f'Simulação {lamb}')
    print(results)
  return results
