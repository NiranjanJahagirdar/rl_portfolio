import numpy as np
import pandas as pd
import matplotlib as plt


#df = pd.read_csv('MQM Q.csv', usecols=[0,1], parse_dates=True, dayfirst=True, index_col=0)
adani = pd.read_csv("/home/niranjan/Individual/ADANIPORTS.csv" , index_col="Date", parse_dates=True)
asian = pd.read_csv("/home/niranjan/Individual/ASIANPAINTS", index_col="Date", parse_dates=True) 
axis = pd.read_csv("/home/niranjan/Individual/AXISBANK", index_col="Date", parse_dates=True) 
bajajauto = pd.read_csv("/home/niranjan/Individual/BAJAJAUTO", index_col="Date", parse_dates=True)  
bajajfs = pd.read_csv("/home/niranjan/Individual/BAJAJFINSV", index_col="Date", parse_dates=True)
bajajfin = pd.read_csv("/home/niranjan/Individual/BAJAJFINANCE", index_col="Date", parse_dates=True) 
airtel = pd.read_csv("/home/niranjan/Individual/BHARTIAIRTEL", index_col="Date", parse_dates=True) 
bpcl = pd.read_csv("/home/niranjan/Individual/BPCL", index_col="Date", parse_dates=True) 
britannia = pd.read_csv("/home/niranjan/Individual/BRITANNIA", index_col="Date", parse_dates=True) 
cipla = pd.read_csv("/home/niranjan/Individual/CIPLA", index_col="Date", parse_dates=True) 
coalind = pd.read_csv("/home/niranjan/Individual/COALINDIA", index_col="Date", parse_dates=True) 
divislab = pd.read_csv("/home/niranjan/Individual/DIVISLAB", index_col="Date", parse_dates=True) 
drreddy = pd.read_csv("/home/niranjan/Individual/DRREDDY", index_col="Date", parse_dates=True) //13
eichermot = pd.read_csv("/home/niranjan/Individual/EICHERMOT", index_col="Date", parse_dates=True) //14
gail = pd.read_csv("/home/niranjan/Individual/GAIL", index_col="Date", parse_dates=True) //15
grasim = pd.read_csv("/home/niranjan/Individual/GRASIM", index_col="Date", parse_dates=True) //16
hcl = pd.read_csv("/home/niranjan/Individual/HCLTECH", index_col="Date", parse_dates=True) //17
hdfc = pd.read_csv("/home/niranjan/Individual/HDFC", index_col="Date", parse_dates=True) //18
hdfcbk = pd.read_csv("/home/niranjan/Individual/HDFCBANK", index_col="Date", parse_dates=True) //19
hdfclife = pd.read_csv("/home/niranjan/Individual/HDFCLIFE", index_col="Date", parse_dates=True) //20
hero = pd.read_csv("/home/niranjan/Individual/HEROMOTOCO", index_col="Date", parse_dates=True) //21
hindalco = pd.read_csv("/home/niranjan/Individual/HINDALCO", index_col="Date", parse_dates=True) //22
hindu = pd.read_csv("/home/niranjan/Individual/HINDUNILVR", index_col="Date", parse_dates=True) //23
icici = pd.read_csv("/home/niranjan/Individual/ICICIBANK", index_col="Date", parse_dates=True) //24
indus = pd.read_csv("/home/niranjan/Individual/INDUSINDBK", index_col="Date", parse_dates=True) //25
infy = pd.read_csv("/home/niranjan/Individual/INFY", index_col="Date", parse_dates=True) //26
ioc = pd.read_csv("/home/niranjan/Individual/IOC", index_col="Date", parse_dates=True) //27
itc = pd.read_csv("/home/niranjan/Individual/ITC", index_col="Date", parse_dates=True) //28
jsw = pd.read_csv("/home/niranjan/Individual/JSWSTEEL", index_col="Date", parse_dates=True) //29
kotak = pd.read_csv("/home/niranjan/Individual/KOTAKBANK", index_col="Date", parse_dates=True) //30
lt = pd.read_csv("/home/niranjan/Individual/LT", index_col="Date", parse_dates=True) //31
mm = pd.read_csv("/home/niranjan/Individual/M&M", index_col="Date", parse_dates=True) //32
maruti = pd.read_csv("/home/niranjan/Individual/MARUTI", index_col="Date", parse_dates=True) //33
nestle = pd.read_csv("/home/niranjan/Individual/NESTLEIND", index_col="Date", parse_dates=True) //34
ntpc = pd.read_csv("/home/niranjan/Individual/NTPC", index_col="Date", parse_dates=True) //35
ongc = pd.read_csv("/home/niranjan/Individual/ONGC", index_col="Date", parse_dates=True) //36
powergrid = pd.read_csv("/home/niranjan/Individual/POWERGRID", index_col="Date", parse_dates=True) //37
rel = pd.read_csv("/home/niranjan/Individual/RELIANCE", index_col="Date", parse_dates=True) //38
sbibk = pd.read_csv("/home/niranjan/Individual/SBIBANK", index_col="Date", parse_dates=True) //39
sbilife = pd.read_csv("/home/niranjan/Individual/SBILIFE", index_col="Date", parse_dates=True) //40
shreecem = pd.read_csv("/home/niranjan/Individual/SHREECEM", index_col="Date", parse_dates=True) //41
sunpharma = pd.read_csv("/home/niranjan/Individual/SUNPHARMA", index_col="Date", parse_dates=True) //42
tatam = pd.read_csv("/home/niranjan/Individual/TATAM", index_col="Date", parse_dates=True) //43
tatas = pd.read_csv("/home/niranjan/Individual/TATASTEEL", index_col="Date", parse_dates=True) //44
tcs = pd.read_csv("/home/niranjan/Individual/TCS", index_col="Date", parse_dates=True) //45
techm = pd.read_csv("/home/niranjan/Individual/TECHM", index_col="Date", parse_dates=True) //46
titan = pd.read_csv("/home/niranjan/Individual/TITAN", index_col="Date", parse_dates=True) //47
ultracem = pd.read_csv("/home/niranjan/Individual/ULTRACEM", index_col="Date", parse_dates=True) //48
upl = pd.read_csv("/home/niranjan/Individual/UPL", index_col="Date", parse_dates=True) //49
wit = pd.read_csv("/home/niranjan/Individual/WIT", index_col="Date", parse_dates=True) //50

stocks = pd.concat([adani,axis,asian,bajajauto,bajajfs,bajajfin,airtel,bpcl,britannia,cipla, coalind, divislab, drreddy, eichermot, gail, grasim, hcl, hdfc, hdfcbk, hdfclife, hero, hindalco, hindu, icici, indus, infy, ioc, itc, lsw, kotak, lt, mm, maruti, nestle, ntpc, ongc, powergrid, rel, sbibk, sbilife, shreecem, sunpharma, tatam, tatas, tcs, techm, titan, ultracem, upl, wit], axis=1)
stock.columns = ["adani","axis","asian","bajajauto","bajajfs","bajajfin","airtel","bpcl","britannia","cipla", "coalind", "divislab", "drreddy","eichermot", "gail", "grasim", "hcl", "hdfc", "hdfcbk", "hdfclife", "hero", "hindalco", "hindu", "icici", "indus", "infy", "ioc", "itc", "lsw", "kotak", "lt", "mm", "maruti", "nestle", "ntpc", "ongc", "powergrid", "rel", "sbibk", "sbilife", "shreecem", "sunpharma", "tatam", "tatas", "tcs", "techm", "titan", "ultracem", "upl", "wit"]

log_ret=np.log(stocks/stocks.shift(1))

np.random.seed(42)
num_ports = 5000
all_weights = np.zeros((num_ports, len(stocks.columns)))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for x in range(num_ports):
    # Weights
    weights = np.array(np.random.random(4))
    weights = weights/np.sum(weights)
    
    # Save weights
    all_weights[x,:] = weights
    
    # Expected return
    ret_arr[x] = np.sum( (log_ret.mean() * weights * 252))
    
    # Expected volatility
    vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*252, weights)))
    
    # Sharpe Ratio
    sharpe_arr[x] = ret_arr[x]/vol_arr[x]
    
print(all_weights[format(sharpe_arr.argmax()):])

plt.figure(figsize=(12,8))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(max_sr_vol, max_sr_ret,c='red', s=50) # red dot
plt.show()

def get_ret_vol_sr(weights):
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov()*252, weights)))
    sr = ret/vol
    return np.array([ret, vol, sr])

def neg_sharpe(weights):
# the number 2 is the sharpe ratio index from the get_ret_vol_sr
    return get_ret_vol_sr(weights)[2] * -1

def check_sum(weights):
    #return 0 if sum of the weights is 1
    return np.sum(weights)-1




