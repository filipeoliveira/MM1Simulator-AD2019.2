import logging
from math import log, sqrt

from random import seed, random
from itertools import count
import time
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(format='[%(levelname)s]: %(message)s', level=logging.INFO)
logging.disable(logging.CRITICAL)

class Statistics(object):
    def __init__(self, id, start_time):

        self.__mean_areas_size = 0
        self.id = id

        self.N_areas_sum = 0
        self.N1_areas_sum = 0
        self.N2_areas_sum = 0
        self.Nq_areas_sum = 0
        self.Nq1_areas_sum = 0
        self.Nq2_areas_sum = 0

        self.mean_N = 0
        self.mean_N1 = 0
        self.mean_N2 = 0
        self.mean_Nq = 0
        self.mean_Nq1 = 0
        self.mean_Nq2 = 0

        self.acumulated_Xr = 0
        self.acumulated_Xr1 = 0
        self.acumulated_Xr2 = 0

        self.mean_Xr = 0
        self.mean_Xr1 = 0
        self.mean_Xr2 = 0

        self.start_time = start_time 

        self.mean_waiting_queue_time = 0
        self.mean_waiting_queue_time_1 = 0  # tempo médio na fila de espera tipo 1
        self.mean_waiting_queue_time_2 = 0  # tempo médio na fila de espera tipo 2

        self.__var_sum_waiting_queue_time_acumulator = 0
        self.__var_len_waiting_queue_time_acumulator = 0
        self.__var_waiting_queue_time_acumulator_sum_of_squares = 0

        self.__var_waiting_queue_time = 0
        self.__var_sum_areas = 0
        self.__var_len_areas = 0
        self.__var_areas_sum_of_squares = 0
        self.__var_areas = 0

        self.waiting_queue_time_acumulator_squared = 0
        self.waiting_queue_time_acumulator_plus_2 = 0
        self.waiting_queue_time_acumulator = 0.0
        self.waiting_queue_time_acumulator_len = 0
        self.waiting_queue_time_acumulator_1 = 0.0
        self.waiting_queue_time_acumulator_1_len = 0 
        self.waiting_queue_time_acumulator_2 = 0.0
        self.waiting_queue_time_acumulator_2_len = 0
        
        self.arrivals_num = 0
        self.total_time = 0.0

    def accumulate_areas(self,area_Nq,area_Nq1,area_Nq2 ,area_N,area_N1,area_N2):
      self.N_areas_sum += area_N
      self.N1_areas_sum += area_N1
      self.N2_areas_sum += area_N2
      self.Nq_areas_sum += area_Nq
      self.Nq1_areas_sum += area_Nq1
      self.Nq2_areas_sum += area_Nq2


    def calculate_mean_w(self):
        self.mean_waiting_queue_time = (self.waiting_queue_time_acumulator) / self.waiting_queue_time_acumulator_len
        return self.mean_waiting_queue_time

    def calculate_mean_w_1(self):
      if not self.waiting_queue_time_acumulator_1_len == 0:
        self.mean_waiting_queue_time_1 = (self.waiting_queue_time_acumulator_1) / self.waiting_queue_time_acumulator_1_len
        return self.mean_waiting_queue_time_1
      else: 
        return 0
    def calculate_mean_w_2(self):
      if not self.waiting_queue_time_acumulator_2_len == 0:
        self.mean_waiting_queue_time_2 = (self.waiting_queue_time_acumulator_2) / self.waiting_queue_time_acumulator_2_len
        return self.mean_waiting_queue_time_2
      else: 
        return 0


    def calculate_mean_N(self, total_round_time):
        self.mean_N = self.N_areas_sum / total_round_time
        self.mean_N1 = self.N1_areas_sum / total_round_time
        self.mean_N2 = self.N2_areas_sum / total_round_time
        self.mean_Nq = self.Nq_areas_sum / total_round_time
        self.mean_Nq1 = self.Nq1_areas_sum / total_round_time
        self.mean_Nq2 = self.Nq2_areas_sum / total_round_time

    def calculate_variance_nq(self, total_round_time):
        mean_squared_acc = nq_square_area_total / total_round_time        
        mean_acc = nq_area_total / total_round_time

        variance_nq = (mean_squared_acc - (mean_acc**2))
        return variance_nq 

    def calculate_t_student_w(self, mean_w, variance_w):
        '''Calculates confidence interval for queue wait time using t-student'''
        # mean_w = self.mean_waiting_queue_time
        # variance_w = self.__var_waiting_queue_time

        lower_limit = mean_w - 1.961 * sqrt(variance_w / ROUNDS_NUM)
        upper_limit = mean_w + 1.961 * sqrt(variance_w / ROUNDS_NUM)
        precision = (upper_limit - lower_limit) / (upper_limit + lower_limit)

        print(f'\n [W] [t-Student] mean_w: {mean_w} | precision: {round(precision*100, 3)}%\n'
              f'Confidence interval: LOWER: {lower_limit} | UPPER: {upper_limit}\n')

        return plot('W', precision, lower_limit, upper_limit, mean_w, variance_w)

    def calculate_t_student_nq(self, mean_nq, variance_nq):
        '''
        Calcula a t-Student do número médio de pessoas na fila de espera.
        Responde parte da letra C do trabalho.
        '''
        lower_limit = mean_nq - 1.961 * sqrt(variance_nq / ROUNDS_NUM)  # t-student table for 0.95 CI and 3200 - 1 DFs
        upper_limit = mean_nq + 1.961 * sqrt(variance_nq / ROUNDS_NUM)  # t-student table for 0.95 CI and 3200 - 1 DFs
        precision = (upper_limit - lower_limit) / (upper_limit + lower_limit)

        print(f'[Nq] [t-Student] mean_nq: {round(mean_nq, 5)} | precision: {round(precision*100, 3)}%\n Confidence interval: LOWER: {lower_limit}'
              f' | UPPER: {upper_limit}\n')

        return plot('NQ', precision, lower_limit, upper_limit, mean_nq, variance_nq)

class plot(object):
    def __init__(self, type, precision, lower_limit, upper_limit, mean, variance):
        self.type = type
        self.precision = precision
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.mean = mean
        self.variance = variance
        self.interval = upper_limit - mean



class Event(object):
    def __init__(self, _type, timestamp, customer_id):
        self.type = _type
        self.timestamp = timestamp
        self.customer_id = customer_id

    def __str__(self):
        return print(f'customer_id: {self.customer_id}\n' \
               f'type: {self.type}\n' \
               f'timestamp: {self.timestamp}\n')

class Customer(object):

    # Initialize a customer with round_number, arrival and service time.
    def __init__(self, id, arrival_time, _type, _mu, _service_type = 'exp', _service_time_limits = [0 , 1]):
      self.id = id
      self.arrival_time = arrival_time
      self.waited_time = 0
      self.departure_time = None
      self.service_start_time = 0
      self.type = _type
      self._mu = _mu
      self._service_type = _service_type
      self._service_time_limits = _service_time_limits

    # Setters
    '''
    O tempo de espera do fregues é dado pelo tempo que ele chega ao sistema até o tempo que ele é servido.
    '''
    def set_service_start_time(self, time_):
      self.service_start_time = time_
      self.set_waited_time()

    def set_departure_time(self, time_):
      self.departure_time = time_

    def set_waited_time(self):
      self.waited_time = self.service_start_time - self.arrival_time

    def reset(self):
      self__ids = count(0)

    def __str__(self):
      return f'id: {self.id}\n' \
              f'arrival_time: {self.arrival_time}\n' \
              f'wait_time: {self.waited_time}\n' \
              f'departure_time: {self.departure_time}\n'
        
class Simulator:
    def __init__(self,counters, datasets, _lambda1, _lambda2, _mu1, _mu2, _priority, _service_type = 'exp', _service_time_limits1 = [5,15], _service_time_limits2 = [1,3] ):
        self.service = 'idle'
        self.serving_customer = None
        self._lambda1 = _lambda1
        self._lambda2 = _lambda2
        self._mu1 = _mu1
        self._mu2 = _mu2
        self._priority = _priority
        self.num_in_system = 0
        self.num_in_system_1 = 0
        self.num_in_system_2 = 0

        self.counters = counters

        self.datasets = datasets
        self.CURRENT_ROUND = 0

        self._service_type = _service_type
        self._service_time_limits1 = _service_time_limits1
        self._service_time_limits2 = _service_time_limits2

        self.clock = 0.0
        self.last_event_time = 0.0

        self.num_arrivals = 0
        self.num_arrivals_1 = 0
        self.num_arrivals_2 = 0
        self.num_departures = 0

        self.customers = []
        self.events = []
        self.waiting_queue = []
        self.priority_queue = []

        # Cria as primeiras chegadas
#        customer = Customer(0.1, 1 , _mu1, _service_type, self._service_time_limits1)
#        customer.reset()
#        print(count(customer.id))
        self.schedule_arrival(1)
        self.schedule_arrival(2)

    def waiting_queue_(self,type):
      size = 0
      for x in self.waiting_queue:
        if x.type == type:
          size += 1
      return size

    def advance_time(self):
        event = self.get_next_event()
        customer_id = event.customer_id

        '''
        Avança o clock até o tempo do próximo relógio.
        Por isso é necessário garantir a ordem cronológica no array de eventos. 
        
        Verifica o tipo de evento e avança o tempo para este evento.      
        '''

        ## Aqui é onde eu acumulo os N, 
        # Tem que ser antes de alterar a fila para coletar o estado anterior dos N, pois o intervalo de tempo que será multiplicado será aquele que acabou de terminar! (werneck)
        self.handle_any_event( customer_id)

        if event.type == 'chegada':
            # f'[Chegada] - clock: {self.clock}, customer_id: {customer_id}'
            logging.info(f'[Chegada] - clock: {self.clock}, customer_id: {customer_id}')    
            self.handle_arrival_event(customer_id)


        elif event.type == 'partida':
            logging.info(f'[Partida] - clock: {self.clock}, customer_id: {customer_id}')
            self.handle_departure_event(customer_id)        



    def get_next_event(self):
        '''
        Retorna o primeiro evento de uma lista de eventos   
        Atualiza o clock para o tempo de processamento do evento           
        Este evento é processado uma única vez.
        '''
        if len(self.events) == 0:
            logging.error("[Exception] - LISTA DE EVENTOS SEM EVENTOS")
            return None

        event = self.events.pop(0)
        self.clock = event.timestamp
        return event


    def handle_any_event(self, customer_id):
      delta_time = self.clock - self.last_event_time
      self.last_event_time = self.clock;

      Nq = len(self.waiting_queue)
      Nq1 = self.waiting_queue_(1)
      Nq2 = self.waiting_queue_(2)
      N  = len(self.waiting_queue) + ( 0 if  self.service == 'idle' else 1 ) 
      N1 = self.waiting_queue_(1) + ( 1 if ( self.service != 'idle' and self.serving_customer.type == 1 ) else 0 )
      N2 = self.waiting_queue_(2) + ( 1 if ( self.service != 'idle' and self.serving_customer.type == 2 ) else 0 )

      area_Nq = Nq * delta_time
      area_Nq1 = Nq1 * delta_time
      area_Nq2 = Nq2 * delta_time
      area_N = N * delta_time
      area_N1 = N1 * delta_time
      area_N2 = N2 * delta_time

      self.datasets[self.CURRENT_ROUND].accumulate_areas(area_Nq,area_Nq1,area_Nq2 ,area_N,area_N1,area_N2)

    def acumulate_Xr(self):  
      next_departure = None
      for event in self.events:
        if event.type == 'partida':
            next_departure = event
      Xr = self.clock - next_departure.timestamp

      self.datasets[self.CURRENT_ROUND].acumulated_Xr += Xr

    def calculate_mean_Xr():
      self.datasets[self.CURRENT_ROUND].mean_Xr = self.datasets[self.CURRENT_ROUND].acumulated_Xr / self.num_arrivals #isso funfa?
      return self.datasets[self.CURRENT_ROUND].mean_Xr

    def pending_service(self): 
      #todo trocar pela média.
      pending = (RHO * self.acumulate_Xr()) /(1 - RHO)
      return pending
      

    def handle_arrival_event(self, customer_id):
      global coletas

      self.num_in_system += 1
      self.num_arrivals += 1

      if self.customers[customer_id].type == 1:
        self.num_in_system_1 += 1
        self.num_arrivals_1 += 1  
      else :
        self.num_in_system_2 += 1
        self.num_arrivals_2 += 1

      logging.info(f"Cliente: {customer_id} do tipo {self.customers[customer_id].type} chegou no sistema | {self.clock} ")
      logging.info(f'Adicionando estatistica do cliente {customer_id} | coleta nº {coletas} \n')
      
      self.datasets[self.CURRENT_ROUND].arrivals_num += 1
      coletas += 1

      self.customers[customer_id].arrival_time = self.clock

      '''
      Quando o servidor estiver desocupado
      => agende um novo evento de chegada com o tempo atual do clock.

      Quando o servidor estiver ocupado
      => adicione o freguês recem chegado à lista de espera. (atenção a política)

      Independentemente da ação
      => agende um novo evento de chegada.
      '''
      if self.service == 'idle':
          self.handle_service_entry(customer_id)
      else:
          self.acumulate_Xr()

          customer = self.customers[customer_id]
          if self._priority:
              if customer.type == 1:
                  self.priority_queue.append(customer)
              else:
                  self.waiting_queue.append(customer)
          else:
              self.waiting_queue.append(customer)

          logging.info(f"Cliente: {customer_id} do tipo {self.customers[customer_id].type} entrou na fila de espera | {self.clock} ")
          for i in self.waiting_queue:
              logging.info(f'Cliente: {i.id} | {i.arrival_time}')
          logging.info('\n')

      self.schedule_arrival(self.customers[customer_id].type)

    def handle_service_entry(self, customer_id):  
        global coletas
        logging.info(f"Cliente: {customer_id} do tipo {self.customers[customer_id].type} entrou em serviço | {self.clock}")
        self.service = 'busy'

        customer = self.customers[customer_id]
        customer.set_service_start_time(self.clock)

        self.serving_customer = customer


        logging.info(f'Adicionando estatistica do cliente {customer_id} | waited_time: {customer.waited_time} '
                     f'| coleta nº {coletas}')
        
        self.datasets[self.CURRENT_ROUND].waiting_queue_time_acumulator += customer.waited_time
        self.datasets[self.CURRENT_ROUND].waiting_queue_time_acumulator_plus_2 += (customer.waited_time * 2) 
        self.datasets[self.CURRENT_ROUND].waiting_queue_time_acumulator_squared += (customer.waited_time * customer.waited_time) 

        self.datasets[self.CURRENT_ROUND].waiting_queue_time_acumulator_len += 1
        
        if customer.type == 1:
          self.datasets[self.CURRENT_ROUND].waiting_queue_time_acumulator_1 += customer.waited_time
          self.datasets[self.CURRENT_ROUND].waiting_queue_time_acumulator_1_len += 1

        else:
          self.datasets[self.CURRENT_ROUND].waiting_queue_time_acumulator_2 += customer.waited_time
          self.datasets[self.CURRENT_ROUND].waiting_queue_time_acumulator_2_len += 1
        
        coletas += 1

        departure_time = self.clock + self.generate_service(customer._mu, customer._service_type, customer._service_time_limits)
        departure = Event('partida', departure_time, customer_id)
        self.schedule(departure)

    def handle_departure_event(self, customer_id):
        self.num_in_system -= 1
        self.num_departures += 1
        self.service = 'idle'
    
        # atualizando informações
        customer = self.customers[customer_id]
        customer.departure_time = self.clock
        self.serving_customer = None

        logging.debug(f"Cliente: {customer_id} do tipo {self.customers[customer_id].type} saiu do sistema | {self.clock}")

        logging.debug("--- Lista de espera: --- ")
        if len(self.waiting_queue) == 0:
            logging.debug("[]")
        else:
            for i in self.waiting_queue:
                logging.debug(f'Cliente: {i.id} | {i.arrival_time}')
        logging.debug('---------------------------')

        '''
        Quando o servidor realizou uma partida e a fila de espera estiver populada
        => agende um novo evento de serviço com o primeiro freguês da fila.
        '''
        if len(self.priority_queue) != 0:
            '''
            If discipline is FCFS next  customer is the first in the queue
            if discipline is LCFS the next customer is the last in queue
            '''
           # if DISCIPLINE == 'FCFS':
            if True:
                customer = self.priority_queue.pop(0)
            else:
                newest_customer = len(self.priority_queue) - 1
                customer = self.priority_queue.pop(newest_customer)

            logging.info(f"Cliente: {customer.id} do tipo {self.customers[customer_id].type}  saiu  da fila de espera | {self.clock} ")
            for i in self.priority_queue:
                logging.debug(f'Cliente: {i.id} | {i.arrival_time}', end=' ')

            self.handle_service_entry(customer.id)
        else:
            if len(self.waiting_queue) != 0:
                '''
                If discipline is FCFS next  customer is the first in the queue
                if discipline is LCFS the next customer is the last in queue
                '''
              #if DISCIPLINE == 'FCFS':
                if True:
                    customer = self.waiting_queue.pop(0)
                else:
                    newest_customer = len(self.waiting_queue) -1
                    customer = self.waiting_queue.pop(newest_customer)

                logging.info(f"Cliente: {customer.id} do tipo {self.customers[customer_id].type}  saiu  da fila de espera | {self.clock} ")
                for i in self.waiting_queue:
                    logging.debug(f'Cliente: {i.id} | {i.arrival_time}', end=' ')

                self.handle_service_entry(customer.id)

    def generate_interarrival(self, _lambda):
        sample = random()
        time_ = log(sample) / (-1 * _lambda )

        return time_

    # @staticmethod
    def generate_service(self, _mu, _service_type = 'exp', serv_time_limits = [0, 1]):
        if _service_type == 'exp':
          sample = random()
          return -1 * log(sample) / _mu
        elif _service_type == 'det':
          return 1 / _mu
        elif _service_type == 'uni':
          [a, b] = serv_time_limits
          sample = random()
          ret = (sample*(b-a))+a
          return 1 / _mu

    def schedule_arrival(self, type):
        '''
        Cria um freguês de acordo com o clock atual acrescido de uma taxa poisson lambda.
        Cria um evento de chegada associado ao freguês criado.
        Agenda tal evento.
        [PS]: caso inicial => self.clock = 0
        '''
        if type == 1:
          arrival_time = self.clock + self.generate_interarrival(self._lambda1)
          _mu = self._mu1
        elif type == 2:
          arrival_time = self.clock + self.generate_interarrival(self._lambda2)
          _mu = self._mu2

        if self._service_type == 'uni':
          if type == 1:
            customer = Customer(self.counters.next_client_id(), arrival_time, type , _mu, self._service_type, self._service_time_limits1)
          elif type == 2:
            customer = Customer(self.counters.next_client_id(),arrival_time, type , _mu, self._service_type, self._service_time_limits2)
        else: 
          customer = Customer(self.counters.next_client_id(),arrival_time, type , _mu, self._service_type)

        self.customers.append(customer)

        event = Event('chegada', customer.arrival_time, customer.id)
        self.schedule(event)


    '''
    Agenda um evento
    Importante ordenar a lista de eventos por ordem cronológica.
    '''
    def schedule(self, event):
        i = 0
        if len(self.events) > 0:  # lista nao vazia
            while i < len(self.events) and self.events[i].timestamp < event.timestamp:
                i += 1
        self.events.insert(i, event)


coletas = 0

class Counters:
  def __init__(self):
    self.client_current_id = 0
    self.statistc_current_id = 0

  def next_client_id(self):
      ret = self.client_current_id
      self.client_current_id = self.client_current_id + 1
      return ret
      
  def next_statistic_id(self):
      ret = self.statistc_current_id
      self.statistc_current_id = self.statistc_current_id + 1
      return ret

counters = None

def run_simulation (_lambda1= 0.1, _lambda2= 0.1, _mu1= 2, _mu2= 2, _service_type = 'exp', _service_time_limits1 = [5, 15], _service_time_limits2 = [1, 3]):
  global coletas
  counters = Counters()
 
  p1 = _lambda1/(_lambda1+_lambda2)
  p2 = _lambda2/(_lambda1+_lambda2)

  RHO = (_lambda1 + _lambda2) * (p1*(1/_mu1) + p2*(1/_mu2) )

  _priority = False

  seed(9707)
  nq_area_total = coletas = nq_square_area_total = 0 
  datasets = []

  mean_w_datasets = []
  mean_w_datasets_1 = []
  mean_w_datasets_2 = []
  mean_nq_datasets = []
  variance_nq_datasets = []

  mean_total_w = 0
  mean_total_w_1 = 0
  mean_total_w_2 = 0
  variance_total_w = 0
  mean_total_nq = 0
  variance_total_nq = 0
  
  # #CENARIOS 1 e 2:
  # _service_type = 'exp'

  # #CENARIO 3
  # _mu1= 1
  # _mu2= 0.5
  # _service_type = 'det'

  # #CENARIO 4 -> parametros da uniforme [ a , b ] do serviço da classe 1 ..
  # _service_time_limits1 = [5, 15]
  # _service_time_limits2 = [1, 3] # e da 2
  # _service_type = 'uni'
  s = Simulator(counters,datasets, _lambda1,_lambda2,_mu1,_mu2,_priority, _service_type, _service_time_limits1, _service_time_limits2)
  start = time.time()


  nq_area_total = 0
  nq_square_area_total = 0
  estimated_variance_nq = 0
  mean_acc_w = 0
  mean_acc_w_1 = 0
  mean_acc_w_2 = 0

  mean_nq = 0

  #Metricas N e NQ
  total_mean_N = total_mean_N1 = total_mean_N2 = 0
  total_mean_Nq = total_mean_Nq1 = total_mean_Nq2 = 0

  total_area_N = total_area_N1 = total_area_N2 = 0
  total_area_Nq = total_area_Nq1 = total_area_Nq2 = 0

  #Métricas W
  accumulated_W = accumulated_W1 = accumulated_W2 = 0
  
  ################################################################################################
  #------------------------------------- SIMULAÇÃO -----------------------------------------------
  ################################################################################################
  for CURRENT_ROUND in range(0, ROUNDS_NUM):
      coletas = 0
      nq_area_total = 0
      nq_square_area_total = 0
      start_time = s.clock
      datasets.append(Statistics(counters.next_statistic_id(), start_time))
      s.CURRENT_ROUND = CURRENT_ROUND

      while coletas < K_SAMPLES:
          s.advance_time()
    
      total_round_time = s.clock - start_time

      datasets[CURRENT_ROUND].calculate_mean_N(total_round_time)

      #TODO printar metricas do round
      total_area_N += datasets[CURRENT_ROUND].mean_N #Array Ni da variancia.
      total_area_N1 += datasets[CURRENT_ROUND].mean_N1
      total_area_N2 += datasets[CURRENT_ROUND].mean_N2
      total_area_Nq += datasets[CURRENT_ROUND].mean_Nq
      total_area_Nq1 += datasets[CURRENT_ROUND].mean_Nq1
      total_area_Nq2 += datasets[CURRENT_ROUND].mean_Nq2    

      # MÉDIA DE CADA RODADA
      accumulated_W += datasets[CURRENT_ROUND].calculate_mean_w()
      accumulated_W1 += datasets[CURRENT_ROUND].calculate_mean_w_1()
      accumulated_W2 += datasets[CURRENT_ROUND].calculate_mean_w_2()

     # print(f'Rodada: {CURRENT_ROUND+1} | RHO: {RHO}  |k_samples: {K_SAMPLES} | mean_w: {accumulated_W/(CURRENT_ROUND+1)} | mean_nq: {datasets[CURRENT_ROUND].mean_Nq}/{CURRENT_ROUND+1} \n')


  total_mean_N = total_area_N/ROUNDS_NUM
  total_mean_N1 = total_area_N1/ROUNDS_NUM
  total_mean_N2 = total_area_N2/ROUNDS_NUM
  total_mean_Nq = total_area_Nq/ROUNDS_NUM
  total_mean_Nq1 = total_area_Nq1/ROUNDS_NUM
  total_mean_Nq2 = total_area_Nq2/ROUNDS_NUM

  #TODO (werneck- jogar dentro de uma func para replicar ela dentro de um for (para cada uma das métricas numéricas - N, Nq, N1, Nq2, etc...) para calcular a variancia)
  sum_variance_Nq = 0
  for CURRENT_ROUND in range(0, ROUNDS_NUM - 1):
    sum_variance_Nq += ((datasets[CURRENT_ROUND].mean_N - total_mean_Nq)**2)

  variance_total_Nq = sum_variance_Nq / (ROUNDS_NUM) # -1 do num de rounds (pq a var começa em 0)

  mean_total_w = accumulated_W/ROUNDS_NUM
  mean_total_w_1 = accumulated_W1/ROUNDS_NUM
  mean_total_w_2 = accumulated_W1/ROUNDS_NUM

  sum_variance_W = 0
  for CURRENT_ROUND in range(0, ROUNDS_NUM -1):
    sum_variance_W += ((datasets[CURRENT_ROUND].calculate_mean_w() - mean_total_w)**2)
   # print(f'sum_variance_W: {sum_variance_W} | total_dividido: {sum_variance_W/(ROUNDS_NUM)}')

  variance_total_W = sum_variance_W / (ROUNDS_NUM)

  print(f'\nSIMULAÇÃO CONCLUIDA | Rodada: {CURRENT_ROUND} | Discipline=[FCFS] k_samples={K_SAMPLES} | RHO= {RHO}')

  print(f'\n------------ SNAPSHOT SERVIDOR ------------\n')
  print(f'clock: {s.clock} | status: {s.service}')
  print(f'arrivals: {s.num_arrivals} | departures: {s.num_departures}')
  print(f'jobs_in_system: {s.num_in_system} | events: {len(s.events)}')
  print(f'jobs_in_waiting_queue: {len(s.waiting_queue)} | arrivals: {s.num_arrivals} | departures: {s.num_departures}')


  print(f'\n-------------- STATISTICS --------------\n')
  print(f'mean_w: {mean_total_w} | variance_w: {variance_total_W}')
  print(f'mean_nq: {total_mean_Nq} | variance_nq: {variance_total_Nq}\n')


  end = time.time()
  ################################################################################################
  #------------------------------------- CONFIDENCE INTERVALS -----------------------------------------------
  ################################################################################################
  # TODO - refatorar o nome com os prints
  tstudent_w = Statistics(0, 0).calculate_t_student_w(mean_total_w, variance_total_W)
  tstudent_nq = Statistics(0, 0).calculate_t_student_nq(total_mean_Nq, variance_total_Nq)

  print(f'Total simulation duration: {round((end - start)/60, 2)} minutes')
  print(f' [{tstudent_w.type}]: precision: {tstudent_w.precision} | lower_limit: {tstudent_w.lower_limit} | upper_limit: {tstudent_w.upper_limit} | mean: {tstudent_w.mean} | variance: {tstudent_w.variance} | interval: {tstudent_w.interval}')

  print(f' [{tstudent_nq.type}]: precision: {tstudent_nq.precision} | lower_limit: {tstudent_nq.lower_limit} | upper_limit: {tstudent_nq.upper_limit} | mean: {tstudent_nq.mean} | variance: {tstudent_nq.variance} | interval: {tstudent_nq.interval} ')
  
  s.customers = []
  resposta = {'w': tstudent_w, 'nq': tstudent_nq} 
  return resposta

ROUNDS_NUM = 1200
K_SAMPLES = 50






# _service_type = 'exp'
  # _service_type = 'det'
# _service_type = 'uni'

#para rodar todos os cenários:
#run_simulation (_lambda1, _lambda2, _mu1, _mu2, _service_type, [5, 15], [1, 3])