import numpy as np
import pandas as pd

from carbontaxdamages.economics import *

show_log = False
def log(msg):
	if show_log:
		print(msg)

# Years used to make the damage function
years = np.arange(2020,2200)


##################################################
##
## Burke growth reductions due to climate change
##
##################################################

# Throughout this script: GMT = Global Mean abolute Temperature

GMT_pre_industrial = 14.0
temperature_2020 = 1.01 # Increase compared to P.I.

def h(GMT, WithLag):
	if WithLag:
		# Regression coefficients for BHM Pooled (inc. 5 years lag)
		beta1, beta2 = -.0037497, -.0000955 
	else:
		# Regression coefficients for BHM Pooled (no lag)
		beta1, beta2 = 0.013036, -0.000496

	return beta1 * GMT + beta2 * GMT**2

def phi(T, WithLag):
	return h(GMT_pre_industrial + T, WithLag) - h(GMT_pre_industrial + temperature_2020, WithLag)





##################################################
##
## Cobb-Douglas economic functions
##
##################################################

alpha = 0.3

def calc_GDP_gross(TFP,L,K):
    return TFP * L**(1-alpha) * K**alpha

def simulation(TFP_values, f, GDP_baseline, population, calibrate=False):

	dk = 0.1
	dt = years[1] - years[0]
	sr = 0.21

	Y_gross, Y = np.zeros(len(years)), np.zeros(len(years))
	if calibrate:
		TFP_values = np.zeros(len(years))

	K = 223.0
	for i, year in enumerate(years):
		if calibrate:
			TFP = GDP_baseline[i] / calc_GDP_gross(1,population[i], K)
			TFP_values[i] = TFP
		
		TFP = TFP_values[i]

		Y_gross[i] = calc_GDP_gross(TFP, population[i], K)
		Y[i] = f[i] * Y_gross[i]

		K = (1-dk)**dt * K + dt * sr * Y[i]

	if calibrate:
		return TFP_values

	return Y_gross, Y





def estimate_phi(f, Y_gross, Y, population, eta):
	Y_gross_PC = Y_gross / population
	Y_PC = Y / population

	estimated_phi = f[1:] * Y_gross_PC[1:] / Y_PC[:-1] - 1 - eta

	return estimated_phi

def estimate_f(true_phi, Y_gross, Y, population, eta):
	Y_gross_PC = Y_gross / population
	Y_PC = Y / population

	estimated_f = np.concatenate((
		[1.0],
		Y_PC[:-1] / Y_gross_PC[1:] * (1 + eta + true_phi)
	))

	return estimated_f





##################################################
##
## Create damage functions
##
##################################################


def create_damage_function(SSP, WithLag):

	### Setup for chosen SSP

	epsilon = 6e-5

	pop = population(years, SSP) * 1e-9
	temp = np.cumsum(baseline_emissions(years[:-1], SSP)) * 0.00062 + temperature_2020
	temp = np.concatenate([[temperature_2020], temp])
	GDP_baseline = GDP(years, SSP)

	# Per capita baseline GDP
	GDP_PC = GDP_baseline / pop * 1e3

	# Calculate TFP by calibrating to SSP GDP
	current_f = np.ones(len(years))
	TFP = simulation([], current_f, GDP_baseline, pop, calibrate=True)

	# Using the baseline temperature and Burke damages, calculate the
	# drop in growth due to climate change (phi) and the growth of
	# GDP without climate change (eta)
	true_phi = phi(temp[:-1], WithLag)
	eta = GDP_PC[1:] / GDP_PC[:-1] - 1


	### Start iterations
	for i in range(500):
		Y_gross, Y = simulation(TFP, current_f, GDP_baseline, pop)

		estimated_phi = estimate_phi(current_f, Y_gross, Y, pop, eta)

		diff = np.sum(np.abs(true_phi - estimated_phi))

		if diff < epsilon:
			log("Converged after {} iterations".format(i))
			break

		if i % 20 == 0:
			log("{}: {}".format(i,diff))

		estimated_f = estimate_f(true_phi, Y_gross, Y, pop, eta)

		current_f = current_f + (estimated_f - current_f) / 2

	return pd.DataFrame({'withlag': int(WithLag), 'SSP': SSP, 'temperature': temp, 'damage': 1-current_f})



# Do all the calculations
all_damages = pd.concat([
	create_damage_function(SSP, WithLag)
	for WithLag in [False, True]
	for SSP in ['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']
])

all_damages.to_csv('carbontaxdamages/data/Burke_damages.csv', index=False)