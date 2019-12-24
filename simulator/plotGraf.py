import matplotlib.pyplot as plt

def plot_E_vs_lambda(mean,lamb1_array,analytical_array,markov_array,simulator,interval,n_layers,pi_matrix_size,name_png):
  title = "Layers: " + str(n_layers) + "    Estates: " + str(pi_matrix_size)
  plt.title(title)

  width = 0.03
  aBar = [] 
  mBar = []
  sBar = []
  fig = plt.figure(figsize=(10, 8))

  for x in range(0,len(lamb1_array)):

    aBar.append(lamb1_array[x] - width/3) 
    mBar.append(lamb1_array[x])
    sBar.append(lamb1_array[x] + width/3) 
    
  plt.xlabel('Lambda')
  plt.ylabel('E['+ mean +']')
  plt.bar(aBar , analytical_array, width/3, label='Analítico')
  plt.bar(mBar,markov_array, width/3,label='CMTC')
  plt.bar(sBar , simulator, width/3, label='Simulação',yerr=interval)

  plt.legend()
  fig.savefig(name_png +'.png')
  plt.show()

  