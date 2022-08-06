# Import libraries
import pandas as pd
import numpy as np
import math
import cplex
import matplotlib.pyplot as plt

def validate(curr_positions, x_optimal, cur_prices, curr_cash):
    
    cash = curr_cash
    portfolio = x_optimal.copy()
    flag = 0
    # iterate until condition met
    while flag == 0:       
        # transaction fee is 0.005 of net transaction therefore use np.absolute
        transaction_fee = 0.005*np.dot(np.absolute(curr_positions - portfolio), cur_prices)
        # by trading we may have postive or negetive amount of cash left
        difference = np.dot((curr_positions - portfolio), cur_prices)
        cash_total = curr_cash + difference
        # if we have sufficient cash to fund the investment,
        # update cash and move on
        if cash_total >= transaction_fee:
            cash = cash_total - transaction_fee
            flag = 1
        else:  # if not, lower num of shares for the most weighted stock
               # and repeat
             cash_gap = abs(cash_total - transaction_fee)
             maximum = 0
             max_index = 0
             for i in range(len(portfolio)):
                 if (portfolio[i]*cur_prices[i]) > maximum:
                     maximum = portfolio[i]
                     max_index = i
             # determine number of shares to drop for the most valued stock
             shares_to_drop = math.ceil(cash_gap/cur_prices[max_index])
             portfolio[max_index] -= shares_to_drop
    return portfolio, cash

# Complete the following functions
def strat_buy_and_hold(x_init, cash_init, mu, Q, cur_prices):
   x_optimal = x_init
   cash_optimal = cash_init
   return x_optimal, cash_optimal

#function for question 3
def strategy_test(x_init, cash_init, mu, Q, cur_prices):
    if np.array_equal(x_init,init_positions): #update weights for the first period
        equal_weight=1/(len(df.columns)-1) #find number of stocks and calculate the weights
        w= np.array(20*[equal_weight]) #equally weighted
        money_available=np.dot(cur_prices, x_init) + cash_init
        num_shares = (money_available * w)/cur_prices
        x_test=np.floor(num_shares)
        cash_test=cash_init
        x_optimal, cash_optimal = validate(x_init,x_test,cur_prices,cash_test)
    else:
        x_optimal=x_init #hold it till the end of period 12
        cash_optimal=cash_init 
    return x_optimal, cash_optimal

def strat_equally_weighted(x_init, cash_init, mu, Q, cur_prices):
    equal_weight=1/(len(df.columns)-1) #find number of stocks and calculate the weights
    w= np.array(20*[equal_weight]) #create array for weights of portfolio
    money_available=np.dot(cur_prices, x_init) + cash_init
    num_shares = (money_available * w)/cur_prices
    x_optimal=np.floor(num_shares)
    cash_optimal=cash_init
    #call validate function to check if strategy is feasible
    return validate(x_init,x_optimal,cur_prices,cash_optimal) 

def strat_min_variance(x_init, cash_init, mu, Q, cur_prices):
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    c  = [0.0] * (len(df.columns)-1)
    lb = [0.0] * (len(df.columns)-1)
    ub = [1.0] * (len(df.columns)-1)
    A = []
    for k in range(len(df.columns)-1):
        A.append([[0],[1.0]])
    var_names = ["w_%s" % i for i in range(1,len(df.columns))]
    cpx.linear_constraints.add(rhs=[1.0], senses="E") 
    cpx.variables.add(obj=c, lb=lb, ub=ub, columns=A, names=var_names)
    Qmat = [[list(range(len(df.columns)-1)), list(2*Q[k,:])] for k in range(len(df.columns)-1)] 
    cpx.objective.set_quadratic(Qmat)
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.solve()
    w=np.array(cpx.solution.get_values())
    money_available=np.dot(cur_prices, x_init) + cash_init
    num_shares = (money_available * w)/cur_prices
    x_optimal=np.floor(num_shares)
    cash_optimal=cash_init    
    #call validate function to check if strategy is feasible
    return validate(x_init,x_optimal,cur_prices,cash_optimal)

def strat_max_Sharpe(x_init, cash_init, mu, Q, cur_prices):
    ##Minimum Variance##
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    c  = [0.0] * (len(df.columns)-1)
    lb = [0.0] * (len(df.columns)-1)
    ub = [1.0] * (len(df.columns)-1)
    A = []
    for k in range(len(df.columns)-1):
        A.append([[0],[1.0]])
    var_names = ["w_%s" % i for i in range(1,len(df.columns))]
    cpx.linear_constraints.add(rhs=[1.0], senses="E") 
    cpx.variables.add(obj=c, lb=lb, ub=ub, columns=A, names=var_names)
    Qmat = [[list(range(len(df.columns)-1)), list(2*Q[k,:])] for k in range(len(df.columns)-1)] 
    cpx.objective.set_quadratic(Qmat)
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.solve()
    w_minVar = np.array(cpx.solution.get_values())
    var_minVar = np.dot(w_minVar, np.dot(Q, w_minVar))
    ret_minVar = np.dot(mu, w_minVar)
    ##Maximize Return##
    cpx2 = cplex.Cplex()
    cpx2.objective.set_sense(cpx2.objective.sense.maximize)
    c2 = []
    for k in range(len(df.columns)-1):
        c2.append(mu[k])
    lb2 = [0.0] * (len(df.columns)-1)
    ub2 = [1.0] * (len(df.columns)-1)
    A2 = []
    for s in range(len(df.columns)-1):
        A2.append([[0],[1.0]])
    var_names2 = ["w_%s" % i for i in range(1,len(df.columns))]
    cpx2.linear_constraints.add(rhs=[1.0], senses="E")
    cpx2.variables.add(obj=c2, lb=lb2, ub=ub2, columns=A2, names=var_names2)
    cpx2.set_results_stream(None)
    cpx2.set_warning_stream(None)
    cpx2.solve()
    w_maxRet = np.array(cpx2.solution.get_values())
    var_maxRet = np.dot(w_maxRet, np.dot(Q, w_maxRet))
    ret_maxRet = np.dot(mu, w_maxRet)
    targetRet = np.linspace(ret_minVar,ret_maxRet,50)
    ##Efficient Frontier##
    cpx3 = cplex.Cplex()
    cpx3.objective.set_sense(cpx3.objective.sense.minimize)
    c3  = [0.0] * (len(df.columns)-1)
    lb3 = [0.0] * (len(df.columns)-1)
    ub3 = [1.0] * (len(df.columns)-1)
    A3 = []
    for t in range(len(df.columns)-1):
        A3.append([[0,1],[1.0,mu[t]]])
    var_names3 = ["w_%s" % i for i in range(1,len(df.columns))]
    cpx3.linear_constraints.add(rhs=[1.0,ret_minVar], senses="EG") 
    cpx3.variables.add(obj=c3, lb=lb3, ub=ub3, columns=A3, names=var_names3)
    Qmat3 = [[list(range(len(df.columns)-1)), list(2*Q[r,:])] for r in range(len(df.columns)-1)] 
    cpx3.objective.set_quadratic(Qmat3)
    cpx3.parameters.threads.set(4)
    cpx3.set_results_stream(None)
    cpx3.set_warning_stream(None)
    w_frontC = []
    var_frontC = []
    ret_frontC = []
    for epsilon in targetRet:
        cpx3.linear_constraints.set_rhs(1,epsilon)
    cpx3.solve()
    w_cur = cpx3.solution.get_values()
    w_frontC.append(w_cur)#weights
    var_frontC.append(np.dot(w_cur, np.dot(Q, w_cur)))
    ret_frontC.append(np.dot(mu, w_cur))
    ##Compute Sharpe Ratio##
    sharpe=(np.array(ret_frontC)-r_rf)/np.sqrt(var_frontC)
    ##find the maximum sharpe ratio##
    ind = np.argmax(sharpe)
    ##return the weight that maximize sharpe ratio##
    weight_array = np.array(w_frontC)
    w = weight_array[ind]
    money_available=np.dot(cur_prices, x_init) + cash_init
    num_shares = (money_available * w)/cur_prices
    x_optimal=np.floor(num_shares)
    cash_optimal=cash_init   
    #call validate function to check if strategy is feasible
    return validate(x_init,x_optimal,cur_prices,cash_optimal)

# Input file
input_file_prices = 'Daily_closing_prices.csv'

# Read data into a dataframe
df = pd.read_csv(input_file_prices)

# Convert dates into array [year month day]
def convert_date_to_array(datestr):
    temp = [int(x) for x in datestr.split('/')]
    return [temp[-1], temp[0], temp[1]]

dates_array = np.array(list(df['Date'].apply(convert_date_to_array)))
data_prices = df.iloc[:, 1:].to_numpy()
dates = np.array(df['Date'])
# Find the number of trading days in Nov-Dec 2019 and
# compute expected return and covariance matrix for period 1
day_ind_start0 = 0
day_ind_end0 = len(np.where(dates_array[:,0]==2019)[0])
cur_returns0 = data_prices[day_ind_start0+1:day_ind_end0,:] / data_prices[day_ind_start0:day_ind_end0-1,:] - 1
mu = np.mean(cur_returns0, axis = 0)
Q = np.cov(cur_returns0.T)

# Remove datapoints for year 2019
data_prices = data_prices[day_ind_end0:,:]
dates_array = dates_array[day_ind_end0:,:]
dates = dates[day_ind_end0:]

# Initial positions in the portfolio
init_positions = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 902, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17500])

# Initial value of the portfolio,V
init_value = np.dot(data_prices[0,:], init_positions)
print('\nInitial portfolio value = $ {}\n'.format(round(init_value, 2)))

# Initial portfolio weights(first row of data_price),w=(v*x)/V
w_init = (data_prices[0,:] * init_positions) / init_value

# Number of periods, assets, trading days
N_periods = 6*len(np.unique(dates_array[:,0])) # =12, with 6 periods per year
N = len(df.columns)-1
N_days = len(dates)

# Annual risk-free rate for years 2020-2021 is 2.5%
r_rf = 0.025

# Number of strategies
strategy_functions = ['strat_buy_and_hold','strat_equally_weighted', 'strat_min_variance', 'strat_max_Sharpe', 'strategy_test']
strategy_names     = ['Buy and Hold', 'Equally Weighted Portfolio', 'Minimum Variance Portfolio', 'Maximum Sharpe Ratio Portfolio','1/n weight and hold']
#N_strat = 1  # comment this in your code
N_strat = len(strategy_functions)  # uncomment this in your code
fh_array = [strat_buy_and_hold, strat_equally_weighted, strat_min_variance, strat_max_Sharpe,strategy_test]

portf_value = [0] * N_strat
#x is a 4x12 matrix, same as cash
#unit to put in each (4 of)strategy for each (12 of) rebalance time
x = np.zeros((N_strat, N_periods),  dtype=np.ndarray)
cash = np.zeros((N_strat, N_periods),  dtype=np.ndarray)
for period in range(1, N_periods+1):
   # Compute current year and month, first and last day of the period
   if dates_array[0, 0] == 20:
       cur_year  = 20 + math.floor(period/7)
   else:
       cur_year  = 2020 + math.floor(period/7)

   cur_month = 2*((period-1)%6) + 1
   day_ind_start = min([i for i, val in enumerate((dates_array[:,0] == cur_year) & (dates_array[:,1] == cur_month)) if val])
   day_ind_end = max([i for i, val in enumerate((dates_array[:,0] == cur_year) & (dates_array[:,1] == cur_month+1)) if val])
   print('\nPeriod {0}: start date {1}, end date {2}'.format(period, dates[day_ind_start], dates[day_ind_end]))
   
   # Prices for the current day
   cur_prices = data_prices[day_ind_start,:]

   # Execute portfolio selection strategies
   for strategy  in range(N_strat):

      # Get current portfolio positions
      if period == 1:
         curr_positions = init_positions
         curr_cash = 0
         portf_value[strategy] = np.zeros((N_days, 1))
      else:
         curr_positions = x[strategy, period-2]
         curr_cash = cash[strategy, period-2]

      # Compute strategy
      x[strategy, period-1], cash[strategy, period-1] = fh_array[strategy](curr_positions, curr_cash, mu, Q, cur_prices)

      # Verify that strategy is feasible (you have enough budget to re-balance portfolio)
      # Check that cash account is >= 0
      # Check that we can buy new portfolio subject to transaction costs

      ###################### Insert your code here ############################
      #Please see the function "validate" above
     
      # Compute portfolio value
      p_values = np.dot(data_prices[day_ind_start:day_ind_end+1,:], x[strategy, period-1]) + cash[strategy, period-1]
      portf_value[strategy][day_ind_start:day_ind_end+1] = np.reshape(p_values, (p_values.size,1))
      print('  Strategy "{0}", value begin = $ {1:.2f}, value end = $ {2:.2f}'.format( strategy_names[strategy], 
             portf_value[strategy][day_ind_start][0], portf_value[strategy][day_ind_end][0]))

      
   # Compute expected returns and covariances for the next period
   cur_returns = data_prices[day_ind_start+1:day_ind_end+1,:] / data_prices[day_ind_start:day_ind_end,:] - 1
   mu = np.mean(cur_returns, axis = 0)
   Q = np.cov(cur_returns.T)
print(cash)
# Plot results
###################### Insert your code here ############################
#Plot one chart illustrates the daily value of your portfolio for each
#trading strategy over the years 2020 and 2021 using daily prices provided.
plt.figure(dpi=1200)
for strategy in range(N_strat): 
    plt.plot(portf_value[strategy][:])
plt.legend(['Buy and hold', 'Equally Weighted', 'Minimum Variance', 'Max SharpeRatio','Equal_weights_hold'], loc='upper left')
plt.show()

#Plot two charts for strategy 3 and 4 to show dynamic changes in portfolio allocations.
strategy3 = np.vstack(x[2,:])
strategy4 = np.vstack(x[3,:])

for n in range(N_periods):
    strategy3[n,:] = strategy3[n,:]/np.sum(strategy3[n,:])
    strategy4[n,:] = strategy4[n,:]/np.sum(strategy4[n,:])
    
plt.figure(dpi=1200)  #high resolution plot  
plt.title('Strategy 3: dynamic changes in portfolio allocations')
color = plt.cm.rainbow(np.linspace(-0.1, 1, 20))
for s, c in zip(range(20), color):
    plt.plot(strategy3[:, s], c=c)
plt.legend(df.columns[1:21], loc='upper left', prop={'size': 5})
plt.xlabel('Time')
plt.ylabel('Weights')
plt.show()

colour = plt.cm.rainbow(np.linspace(0.04, 0.9, 20)) 
plt.figure(dpi=1200) 
plt.title('Strategy 4: dynamic changes in portfolio allocations')
for t,d in zip(range(20),colour):
    plt.plot(strategy4[:, t],c=d)
plt.legend(df.columns[1:21], loc='upper left', prop={'size': 5})
plt.xlabel('Time')
plt.ylabel('Weights')
plt.show() 