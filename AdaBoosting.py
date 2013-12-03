#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  AdaBoosting.py
#  
#  Copyright 2013 Phiradet Bangcharoensap <phiradet@gmail.com>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import numpy as np
import math

class AdaBoost:
	def __init__(self):
		self.distribution = []
		self.hypothesis = []
		self.alpha = []
	
	def genHypothesis(self, i):
		return lambda x: x[i]
	
	def partialPredict(self, hypothesis, samples):
		return [hypothesis(x) for x in samples];
		
	def calR(self, h_t, labels):
		isError = np.zeros(len(h_t))
		for i in range(len(h_t)):
			if h_t[i]!=labels[i]:
				isError[i]=1;
		r_t = 0.5-(isError*self.distribution).sum()
		return r_t
		
	def sign(self, x):
		return -1 if x<0 else 1;
	
	def train(self, samples, labels, T, acceptError):
		self.m = len(samples)
		self.distribution = np.ones(self.m)/self.m
		for t in range(T):
			currHypothesis = self.genHypothesis(t)
			self.hypothesis.append(currHypothesis)
			h_t = self.partialPredict(currHypothesis, samples)
			r_t = self.calR(h_t, labels)
			beta = math.sqrt((1.0-2.0*r_t)/((1.0+2.0*r_t)+1e-6))		
			alpha = math.log(1.0/(beta+1e-6))
			self.alpha.append(alpha)
			
			f_t = lambda x: self.sign(sum([self.alpha[i]*self.hypothesis[i](x) for i in range(len(self.hypothesis))]))
			isCorrect = []
			
			currPredictError = []
			for i in range(self.m):
				predict_out = f_t(samples[i])
				if predict_out != labels[i]:
					currPredictError.append(1.0)
				else:
					currPredictError.append(0.0)
					
			errorProb = sum(currPredictError)/self.m
			print 'iter',t,errorProb
			if errorProb<acceptError:
				break
			sumDistribution = 0.0
			for i in range(len(h_t)):
				if h_t[i]!=labels[i]:
					self.distribution[i] = self.distribution[i]*(1.0/(beta+1e-6))
				else:
					self.distribution[i] = self.distribution[i]*(beta)
				sumDistribution+=self.distribution[i]
			self.distribution=self.distribution/sumDistribution

		#print self.alpha;

	def predict(self, sample):
		predictSum = 0
		for i in range(len(self.hypothesis)):
			predictSum += self.alpha[i]*self.hypothesis[i](sample)
		return self.sign(predictSum)

def readInput(filename="mushroomB5000.txt"):
	f = open(filename, 'r')
	lines = f.readlines()
	f.close()
	labels = []
	samples = []
	for line in lines:
		tokens = line.strip().split(' ')
		if tokens[0]=='-1':
			break
		labels.append(-1 if tokens[0]=='0' else 1)	
		samples.append(map(lambda x:-1 if x=='0' else 1, tokens[1:]))
	return samples, labels

def evaluation(obj, filename="mushroomB3000.txt"):
	samples, labels = readInput(filename=filename)
	TP = 0.0
	FP = 0.0
	TN = 0.0
	FN = 0.0
	for i in range(len(samples)):
		currSample = samples[i]
		currLabel = labels[i]
		predictResult = obj.predict(currSample)
		if predictResult == 1:
			if predictResult==currLabel:
				TP+=1.0
			else:
				FP+=1.0
		else:
			if predictResult==currLabel:
				TN+=1.0
			else:
				FN+=1.0
	print "--- Confusion Matrix ---"
	print TP, FP
	print FN, TN 
	print "------------------------"
	print "accuracy: %.2f %%"%((TP+TN)/(TP+TN+FP+FN)*100)
	print "sensitivity: %.2f %%"%((TP)/(TP+FN)*100)
	print "specificity: %.2f %%"%((TN)/(FP+TN)*100)
			
def main():
	samples, labels = readInput()
	print len(samples[0])
	ada = AdaBoost()
	ada.train(samples, labels, 119, 0.01)
	evaluation(ada)
	return 0

if __name__ == '__main__':
	main()

