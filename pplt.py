
import numpy as np
import matplotlib.pyplot as plt

if True: # sgd gradual
    w0 = {'FTA': 1, 'FTF': 1,'L2SP':2, 'ST': 2, 'MMTL': 18}
    w1 = {'FTA': 2, 'FTF': 1,'L2SP':2, 'ST': 1, 'MMTL': 17}
    w2 = {'FTA': 3, 'FTF': 1,'L2SP':1, 'ST': 0, 'MMTL': 19}
    w3 = {'FTA': 8, 'FTF': 0,'L2SP':0, 'ST': 0, 'MMTL': 14}
    t =  "SGD MMTL vs ALL"
    cc = 'g'
    loc = 'upper right'
elif False:
    w0 = {'FTA': 2, 'FTF': 2,'L2SP':5, 'ST': 8, 'OCTL': 8}
    w1 = {'FTA': 4, 'FTF': 1,'L2SP':3, 'ST': 4, 'OCTL': 11}
    w2 = {'FTA': 3, 'FTF': 0,'L2SP':1, 'ST': 1, 'OCTL': 16}
    w3 = {'FTA': 0, 'FTF': 0,'L2SP':0, 'ST': 0, 'OCTL': 20}
    t =  "SGD OCTL vs ALL"
    cc = 'limegreen'
    loc = 'upper left'
elif False:
    w0={'FTA': 0, 'ST': 3, 'GDA': 19, 'FTF': 1}
    w1 = {'FTA': 0, 'ST': 2, 'GDA': 17, 'FTF': 2}
    w2 = {'FTA': 0, 'ST': 1, 'GDA': 16, 'FTF': 3}
    w3 = {'FTA': 0, 'ST': 3, 'GDA': 16, 'FTF': 4}
    t =  "Adam GDA vs ALL"
    cc = 'green'
    loc = 'upper right'
elif True:
    w0 = {'FTA': 0, 'ST': 13, 'CO': 1, 'FTF': 9}
    w1 = {'FTA': 0, 'ST': 10, 'CO': 1, 'FTF': 10}
    w2 = {'FTA': 0, 'ST': 8, 'CO': 2, 'FTF': 10}
    w3 = {'FTA': 3, 'ST': 1, 'CO': 5, 'FTF': 14}
    t =  "Adam CO vs ALL"
    cc = 'limegreen'
    loc = 'upper left'
ws = [w0,w1,w2,w3]
for w in ws:
    ss = sum(w.values())
    for k,v in w.items():
        w[k] = int(v/ ss *30)
N = 4
ind = np.array([0,1.25,2.5,3.75])
width = 0.25
bars = []
descs =[]
for k,c,w in zip(w0.keys(),['r','b','k','y',cc],[0,0.2,0.4,0.6,0.8]):
    xvals = [w[k]+0.1 for w in ws ]

    bar1 = plt.bar(ind+w, xvals, width, color =c)
    bars.append(bar1)
    descs.append(k)

plt.xlabel("num. of slices",fontsize=14)
plt.ylabel('num. of pairs',fontsize=14)
#
# plt.title(t)

plt.xticks(ind+width,['90', '270', '540','1080'])
plt.yticks(range(0,35,5))
plt.legend( bars,descs,loc=loc )
plt.savefig('xxxxxxxxxx'+t+'.png',bbox_inches='tight')
plt.show()
