import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('network_backup_dataset.csv')
data.columns = ['week', 'day_of_week', 'start_time','work_flow','file_name_str','size','time']


data['day_of_week']=data['day_of_week'].replace(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'], 
                                [1,2,3,4,5,6,7])

data['work_flow']=data['work_flow'].replace(['work_flow_0','work_flow_1','work_flow_2','work_flow_3','work_flow_4'], 
                                [0,1,2,3,4])

# add day column
day = []
for i in range(0, len(data)):
    day.append((data['week'][i]-1)*7 + data['day_of_week'][i])
data.insert(2, 'day', day)

# 20 days period
period = 20
# 105 days period
#period = 105
sum_size = {0:[0.0]*period, 1:[0.0]*period, 2:[0.0]*period, 3:[0.0]*period, 4:[0.0]*period}
for i in range(0, len(data)):
    if day[i] > period:
        break
    sum_size[data['work_flow'][i]][day[i]-1] += data['size'][i]


plt.figure(figsize = (10,8))
# 105 days period
# plt.figure(figsize = (15,8))
color = ['b', 'k', 'r', 'm', 'g']
for i in range(0, 5):
    plt.plot(range(1, period+1), sum_size[i], linewidth=1,  c = color[i], label = 'work_flow_'+str(i))

plt.ylabel('Total Backup Size (GB)',fontsize=15)
plt.xlabel('Days',fontsize=15)
plt.title('Total Backup Size   VS   days',fontsize=15)
plt.axis([0,period+1,0,12])
plt.legend(loc = 'upper right', fontsize=15)
plt.show()


