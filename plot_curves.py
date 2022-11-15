#!/usr/bin/env python
# coding: utf-8

# In[152]:


import numpy as np
import matplotlib.pyplot as plt
from math import floor, log10


# In[71]:


dt = np.load('convergence_report.npy', allow_pickle = True).item()
dt


# In[183]:


#dp and dq

dpi = 400
fig, ax = plt.subplots(dpi = dpi)
al = 1
color = ['blue','red', 'darkgreen', 'orange',]
list_name = ['$|\Delta P|$', '$|\Delta Q|$']

dP = np.linalg.norm(dt['dP'], axis = 1)
dQ = np.linalg.norm(dt['dQ'], axis = 1)
x = np.arange(0, dt['dP'].shape[0])

ax.plot(x, dP, color = 'red', alpha = al, label = list_name[0], marker = 'o')
ax.plot(x, dQ, color = 'blue', alpha = al, label = list_name[1], marker = 'o')

ax.set_yscale('log')
plt.legend(frameon = False)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
#ax.spines['left'].set_smart_bounds(True)
#ax.spines['bottom'].set_smart_bounds(True)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks(np.arange(0, dt['dQ'].shape[0]))
plt.xlabel('Iteração')
plt.ylabel('$|\Delta X|$')

ax.axhline(dt['error'], ls = '-.', color = 'green', alpha = 1, lw = .7)
exp = floor(np.log10(dt['error']))
base = dt['error']*10**(-exp)
ax.text(0, dt['error'] + 0.5e-4, '$\epsilon_{max} = %.1f\\times10^{%i}$' %(base, exp), color = 'green', size = 'small')
plt.savefig('deltaPQ.png')


# In[189]:


#V

dpi = 400
fig, ax = plt.subplots(dpi = dpi)
al = .5
color = ['blue','red', 'darkgreen', 'orange',]
list_name = ['Swing', '$PV_1$', '$PQ_1$', '$PQ_2$']
list_name = ['Barra 4 (Swing)', 'Barra 1 (PV)', 'Barra 2 (PQ)', 'Barra 3 (PQ)']

V = np.transpose(dt['V'])
x = np.arange(0, dt['V'].shape[0])


ax.plot(x, V[1], color = 'blue', alpha = al, label = list_name[1], marker = '^')
ax.plot(x, V[2], color = 'green', alpha = al, label = list_name[2], marker = 'o')
ax.plot(x, V[3], color = 'orange', alpha = al, label = list_name[3], marker = 'o')
ax.plot(x, V[0], color = 'red', alpha = al, label = list_name[0], marker = 'v') #Swing

plt.legend(loc = 'lower right', frameon = False)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
#ax.spines['left'].set_smart_bounds(True)
#ax.spines['bottom'].set_smart_bounds(True)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks(np.arange(0, dt['dQ'].shape[0]))
plt.xlabel('Iteração')
plt.ylabel('$V_{pu}$')
plt.savefig('V.png')

#ax.axhline(dt['error'], ls = '-.', color = 'green', alpha = 1, lw = .7)
#exp = floor(np.log10(dt['error']))
#base = dt['error']*10**(-exp)
#ax.text(0, dt['error'] + 0.5e-4, '$\epsilon_{max} = %.1f\\times10^{%i}$' %(base, exp), color = 'green', size = 'small')


# In[190]:


#theta

dpi = 400
fig, ax = plt.subplots(dpi = dpi)
al = .5
color = ['blue','red', 'darkgreen', 'orange',]
list_name = ['Swing', '$PV_1$', '$PQ_1$', '$PQ_2$']
list_name = ['Barra 4 (Swing)', 'Barra 1 (PV)', 'Barra 2 (PQ)', 'Barra 3 (PQ)']


V = 180*np.transpose(dt['theta'])/np.pi
x = np.arange(0, dt['theta'].shape[0])


ax.plot(x, V[1], color = 'blue', alpha = al, label = list_name[1], marker = '^')
ax.plot(x, V[2], color = 'green', alpha = al, label = list_name[2], marker = 'o')
ax.plot(x, V[3], color = 'orange', alpha = al, label = list_name[3], marker = 'o')
ax.plot(x, V[0], color = 'red', alpha = al, label = list_name[0], marker = 'v') #Swing

plt.legend(loc = 'upper right', frameon = False)

ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
#ax.spines['left'].set_smart_bounds(True)
#ax.spines['bottom'].set_smart_bounds(True)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.set_xticks(np.arange(0, dt['dQ'].shape[0]))
plt.xlabel('Iteração')
plt.ylabel('$\\theta (°)$')

plt.savefig('theta.png')

