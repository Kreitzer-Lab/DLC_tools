import pandas as pd 
import numpy as np  
from scipy.spatial.distance import pdist, squareform


# Returns displacement dataframe given X & Y:
def get_displacement(df_x,df_y):
	dx = (df_x.shift(1)-df_x)**2
	dy = (df_y.shift(1)-df_y)**2
	disp = (dx.add(dy))**0.5
	return disp


class DataCleaner:
	def __init__(self,filename):
		df = pd.read_csv(filename,header=[0,1,2])	
		self.df_orig = df
		body_parts = [df.columns[idx][1] for idx in range(1,df.shape[1],3)]

		self.body_parts = body_parts

		# confidence values
		self.conf = df.iloc[:,range(3,df.shape[1],3)]
		self.conf.columns = body_parts

		# Position:
		self.x = df.iloc[:,range(1,df.shape[1],3)]
		self.x.columns = body_parts
		self.y = df.iloc[:,range(2,df.shape[1],3)]
		self.y.columns = body_parts

		# Displacement:
		self.disp = get_displacement(self.x,self.y)



	def interpolate(self):
		self.x = self.x.interpolate()
		self.y = self.y.interpolate()
		self.disp = get_displacement(self.x,self.y)


	def remove_low_likelihood(self,threshold):
		# Want to be able to try several versions of this
		self.x = self.x.mask(self.conf<threshold)
		self.y = self.y.mask(self.conf<threshold)
		self.disp = get_displacement(self.x,self.y)


	def remove_jumps(self,threshold):
		disp = self.disp.to_numpy()
		x = self.x.to_numpy()
		y = self.y.to_numpy()
		for frame in range(1,self.disp.shape[0]):
			chg_idx = np.where(disp[frame]>threshold)[0]
			x[frame,chg_idx] = np.nan
			y[frame,chg_idx] = np.nan
		self.x[self.body_parts] = x
		self.y[self.body_parts] = y
		self.disp = get_displacement(self.x,self.y)


	def remove_body_swaps(self):
		t0 = np.ndarray(shape=(self.x.shape[0],2,6)) # (time,coord,part)
		t0[:,0,:] = self.x.to_numpy()
		t0[:,1,:] = self.y.to_numpy()
		t_copy = np.copy(t0)
		t_new = np.copy(t0)
		for frame in range(1,t0.shape[0]):
		    k = squareform(pdist(np.concatenate((t0[frame,:,:],t_copy[frame-1,:,:]),axis=1).T))
		    min_index = np.argmin(k[6:,0:6],axis=0) # minimum index body part 
		    chng_idx = np.where(min_index-np.arange(0,6)!=0)[0] # gets index of body part to change
		    t0[frame,:,chng_idx] = t0[frame-1,:,chng_idx] # set to previous frame's value
		    t_new[frame,:,chng_idx] = np.nan # set to NaN if detected     
		self.x = pd.DataFrame(t_new[:,0,:],columns=self.body_parts)
		self.y = pd.DataFrame(t_new[:,1,:],columns=self.body_parts)
		self.disp = get_displacement(self.x,self.y)



	def write_csv(self,filename):
		# couldn't figure out how to avoid these stupid loops:
		for i,x in enumerate(range(1,self.conf.shape[1],3)):
			self.df_orig.iloc[:,x] = self.x.iloc[:,i]
		for i,x in enumerate(range(2,self.conf.shape[1],3)):
			self.df_orig.iloc[:,x] = self.y.iloc[:,i]
		self.df_orig.to_csv(filename)





	# Want something to show distributions of 

