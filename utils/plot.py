'''
Created on Sep 24, 2013

@author: luamct
'''
from collections import defaultdict  
import matplotlib.pyplot as pp
import numpy as np

import matplotlib
import random

# Changing some default configurations
matplotlib.rc('font', size=15)
# matplotlib.rc('figure', figsize=(8,6))


def show() :

	def close_figure(event):
		if event.key == 'escape' :
			pp.close(event.canvas.figure)

	pp.gcf().canvas.mpl_connect('key_press_event', close_figure)
	pp.show()


def count(values):
	'''
	Converts a list of values into a dict of frequencies (histogram)
	'''
	c = defaultdict(int)
	for v in values :
		c[v] += 1
	return c


def cdf(v, title='', 
				xlabel='', ylabel='', 
				xlim=(), ylim=(), 
				xscale='linear', yscale='linear', 
				linewidth=1.5, 
				outfile=None) :

	fs = count(v)
	values, freqs = zip(*sorted(fs.items()))  # Split values and frequencies sorting by the values

	cum  = np.cumsum(freqs, dtype=np.float64)
	cum /= np.sum(freqs)

	pp.clf()
	
	matplotlib.rc('font', size=24)
	
	pp.title(title) #, {'fontsize' : 22}
	pp.xlabel(xlabel)
	pp.ylabel(ylabel)
	pp.xscale(xscale)
	pp.yscale(yscale)
	pp.grid()
# 	pp.tight_layout(pad=0.2)
	
# 	pp.yscale('log')
	
	if xlim : pp.xlim(xlim)
	if ylim : pp.ylim(ylim)

	pp.tight_layout(pad=0.10)
	pp.plot(values, cum, lw=linewidth)
#	pp.show()
	
	if outfile:
		pp.savefig(outfile)
	
	

def rank(v, title='', xlabel='', ylabel='', xlim=(), ylim=(), xscale='linear', yscale='linear', linewidth=2, outfile='') :

	v.sort(reverse=True)

	pp.clf()
	pp.title(title)
	pp.xlabel(xlabel)
	pp.ylabel(ylabel)
	pp.xscale(xscale)
	pp.yscale(yscale)
	pp.grid()
	
	
	if xlim : pp.xlim(xlim)
	if ylim : pp.ylim(ylim)

	# Remove zeros
	v = filter(lambda x: x>0, v)

	cum =  np.cumsum(v, dtype=np.float64)
	cum /= cum[-1]
	
	pp.plot(np.arange(1, len(cum)+1), cum, lw=linewidth)
	
# 	pp.plot(values, cum, lw=linewidth)
	if outfile:
		pp.savefig(outfile)
		
	else:
		show()
	

def scatter(x, y, ids, 
						title='', 
					  xlabel='', ylabel='', 
					  xlim=(), ylim=(), 
					  xscale='linear', yscale='linear', 
					  stratify=0,
					  pointsize=1, 
					  outfile='') :

	pp.clf()
# 	pp.title(title)
	pp.xlabel(xlabel)
	pp.ylabel(ylabel)
	pp.xscale(xscale)
	pp.yscale(yscale)
	
# 	pp.yscale('log')
	
	if xlim : pp.xlim(xlim)
	if ylim : pp.ylim(ylim)

	if stratify>0 and ylim :
		
		sample_size = 50
		stratified = []
		data = sorted(zip(y, x))
		bin_range = (ylim[1] - ylim[0])/stratify
		
		b = 1
		last = 0
		for i, (y, x) in enumerate(data) :

			if y > b*bin_range :
				print "%d : %d => %d " %((b-1)*bin_range, (b)*bin_range, i-last)
				stratified.extend(random.sample(data[last:i], min(i-last,sample_size)))
				last = i
				b += 1
				
		
		y, x = zip(*stratified)
		

	def onpick(event):

		N = len(event.ind)
		if not N: return True

		for point in event.ind:
			print "%s : (%f, %f)" % (ids[point], x[point], y[point])

		return True


	pp.plot(x, y, 'o', picker=6)

	# Either save the plot or show it
	if outfile:
		pp.savefig(outfile)
	
	else:
		pp.gcf().canvas.mpl_connect('pick_event', onpick)
		show()



def bars(x, values, 
				width=1.0, 
				bin_count=[], 
				err=(), 
				ticklabels=[],
				title='', 
				xlabel='', ylabel='', 
				xlim=(), ylim=(), 
				xscale='linear', yscale='linear',
				pointsize=1, 
				outfile='') :

	# Set all given arguments
	pp.clf()
	pp.figure(figsize=(9,5))
	pp.title(title)

	pp.xlabel(xlabel)
	pp.ylabel(ylabel)
	pp.xscale(xscale)
	pp.yscale(yscale)

	x = np.array(x)

	if len(err) :
		pp.errorbar(x+0.5*width, values, err, fmt='.', elinewidth=1, ecolor=(0.7,0.1,0), ms=0, capthick=0.7, capsize=3)

	if len(ticklabels) :
		pp.gca().set_xticks(x+0.5*width)
		pp.gca().set_xticklabels(ticklabels, rotation=45, fontsize=13)

	# Plot size of each bin above the bars
	offset = 0.05*np.mean(values)
	for b in xrange(len(bin_count)) :
		pp.text(x[b]+0.3*width, values[b]+offset, str(bin_count[b]), {'size': 7}, rotation=90, ha='center', va='bottom')

	# Bottom is zero, but top is left for matplot to decide	
	pp.ylim(bottom=0)

	if xlim : pp.xlim(xlim)
	if ylim : pp.ylim(ylim)
	
	pp.bar(x, values, width, linewidth=1.2, color='#778899')
	
	pp.axhline(0, color='black', lw=1.4)
	
	pp.tight_layout()
	if outfile:
		pp.savefig(outfile)
	else: 
		show()


def lines(x, ys, legends, xlabel='', ylabel='', xlim=(), ylim=(), title='', outfile='') :
	
	pp.figure()
	pp.title(title)
	pp.xlabel(xlabel)
	pp.ylabel(ylabel)
	
	if xlim : pp.xlim(xlim)
	if ylim : pp.ylim(ylim)
	
	markers = ['-s', '-o', '-^', '-h', '-*']
	for i in xrange(len(ys)): 
		pp.plot(x, ys[i], markers[i], label=legends[i], lw=1.5, ms=8.0)

	pp.legend(fontsize="medium", loc="best")
	
	if outfile : 
		pp.savefig(outfile)
	pp.show() 
	

if __name__ == "__main__" :
	
# 	x1 = np.arange(0.0, 2.0, 0.1)
# 	y1 = 2*x1
# 	y2 = x1**2
# 	y3 = 3.6*np.sin(x1)
# 	lines((x1, x1, x1), (y1, y2, y3), ["Social", "Aesthetics", "Semantics"], xlabel="Gap (%)", ylabel="Accuracy")

	x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	aes_y = [0.5363, 0.5408, 0.5440, 0.5500, 0.5531, 0.5571, 0.5639, 0.5714, 0.5808, 0.5877]
	sem_y = [0.6514,0.6657,0.6782,0.6938,0.7052,0.7141,0.7203,0.7278,0.7272,0.7429]
	soc_y = [0.7659, 0.7993, 0.8300, 0.8577, 0.8810, 0.8996, 0.9150, 0.9320, 0.9484, 0.9575]
	
	lines((x, x, x), 
				(aes_y, sem_y, soc_y), 
				["Aesthetics", "Semantics", "Social"], 
				xlabel="Gap (%)", 
				ylabel="Accuracy",
				outfile="accuray.pdf")
	
	
	
	