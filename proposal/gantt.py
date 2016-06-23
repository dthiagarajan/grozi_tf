"""
Creates a simple Gantt chart
Adapted from http://www.clowersresearch.com/main/gantt-charts-in-matplotlib/
"""
 
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib.dates
from matplotlib.dates import WEEKLY, DateFormatter, rrulewrapper, RRuleLocator
 
from pylab import *
 
def create_date(month, day):
	"""Creates the date"""
	date = dt.datetime(2016, int(month), int(day))
	mdate = matplotlib.dates.date2num(date)
	return mdate
 
# Data
 
pos = arange(0.5,6.5,0.5)

ylabels = [
	'Basic network, single class',
	'ROC curve',
	'Small network from scratch',
	'Fine-tune large network',
	'Performance characteristics',
	'Train with 120 categories',
	'Localization: add RPN',
	'Pipeline to feed recorded video',
	"Record Mattin's footage",
	'Real-time application w/ camera',
	'Speed/accuracy improvements',
	'Continuity report'
]

customDates = [
	[(6,13), (6,20)],
	[(6,13), (6,20)],
	[(6,20), (6,27)],
	[(6,20), (6,27)],
	[(6,24), (7,1)],
	[(6,24), (7,4)],
	[(7,4), (7,13)],
	[(7,13), (7,25)],
	[(7,25), (8,12)],
	[(7,25), (8,12)],
	[(8,1), (8,12)],
	[(7,4), (8,19)],
]
customDates = map(lambda lst: map(lambda x: create_date(x[0], x[1]), lst),
				  customDates)

task_dates = {}
for i, task in enumerate(ylabels):
	task_dates[task] = customDates[i]
 
# Initialise plot
 
fig = plt.figure()
ax = fig.add_axes([0.25,0.1,0.7,0.8]) #[left,bottom,width,height]
ax.set_title('Project Timeline')
# ax = fig.add_subplot(111)
 
# Plot the data
 
for i in range(len(ylabels)):
	start_date, end_date = task_dates[ylabels[i]]
	ax.barh((i+1) * 0.5, end_date-start_date, left=start_date, height=0.3, align='center', color='blue')
# start_date,end_date = task_dates[ylabels[0]]
# ax.barh(0.5, end_date - start_date, left=start_date, height=0.3, align='center', color='blue', alpha = 0.75)
# # ax.barh(0.45, (end_date - start_date)*effort[0][0], left=start_date, height=0.1, align='center', color='red', alpha = 0.75, label = "PI Effort")
# # ax.barh(0.55, (end_date - start_date)*effort[0][1], left=start_date, height=0.1, align='center', color='yellow', alpha = 0.75, label = "Student Effort")
# for i in range(0,len(ylabels)-1):
# 	# labels = ['Analysis','Reporting'] if i == 1 else [None,None]
# 	start_date,mid_date,end_date = task_dates[ylabels[i+1]]
# 	# piEffort, studentEffort = effort[i+1]
# 	ax.barh((i*0.5)+1.0, mid_date - start_date, left=start_date, height=0.3, align='center', color='blue', alpha = 0.75)
# 	# ax.barh((i*0.5)+1.0-0.05, (mid_date - start_date)*piEffort, left=start_date, height=0.1, align='center', color='red', alpha = 0.75)
# 	# ax.barh((i*0.5)+1.0+0.05, (mid_date - start_date)*studentEffort, left=start_date, height=0.1, align='center', color='yellow', alpha = 0.75)
# 	# ax.barh((i*0.5)+1.0, end_date - mid_date, left=mid_date, height=0.3, align='center',label=labels[1], color='yellow')
 
# Format the y-axis
 
locsy, labelsy = yticks(pos,ylabels)
plt.setp(labelsy, fontsize = 14)
 
# Format the x-axis
 
ax.axis('tight')
ax.set_ylim(ymin = -0.1, ymax = 6.5)
ax.grid(color = 'g', linestyle = ':')
 
ax.xaxis_date() #Tell matplotlib that these are dates...
 
rule = rrulewrapper(WEEKLY, interval=1)
loc = RRuleLocator(rule)
formatter = DateFormatter("%b %d")
 
ax.xaxis.set_major_locator(loc)
ax.xaxis.set_major_formatter(formatter)
labelsx = ax.get_xticklabels()
plt.setp(labelsx, rotation=30, fontsize=12)
 
# Format the legend
 
# font = font_manager.FontProperties(size='small')
# ax.legend(loc=1,prop=font)
 
# Finish up
ax.invert_yaxis()
fig.autofmt_xdate()
#plt.savefig('gantt.svg')
plt.show()