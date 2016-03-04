from __future__ import print_function
from __future__ import division

'''
Random state has yet to be fully implemented. Otherwise it is working!
'''
__author__ = 'fernando nogueira'


import numpy
from datetime import datetime

from scipy.optimize import basinhopping
from scipy.optimize import minimize
from scipy.stats import norm
from math import sqrt

from sklearn.cross_validation import cross_val_score
from sklearn.gaussian_process import GaussianProcess


class BayesianOptCV:


		# --------------------------------------------- // --------------------------------------------- #
		# --------------------------------------------- // --------------------------------------------- #
		def __init__(self, estimator, \
								 param_bounds, param_types = None, param_fixed = None, param_list = None, \
								 acq = 'ei', gp_params = None,\
								 n_iter=10, scoring=None, fit_params=None, cv=None,\
								 n_jobs=1, verbose=0, random_state=None):


				self.estimator = estimator

				self.pbounds = param_bounds
				self.pfixed = param_fixed
				self.plist = param_list

				self.ptypes = {key: type(v1) for key, (v1, _v2_) in param_bounds.items()}

				if param_list != None:
						for key in param_list.keys():
								self.pbounds[key] = (1, len(param_list[key]) + 0.9999999)

				self.acq = acq
				self.gp_params = gp_params

				self.niter = n_iter

				self.cv_params = {'estimator' : estimator,\
													'X' : None,\
													'y' : None,
													'scoring' : scoring,\
													'fit_params' : fit_params,
													'n_jobs' : n_jobs,\
													'cv' : cv,\
													'verbose' : verbose,\
													'pre_dispatch' : 2*n_jobs}

				self.random_state = random_state

				self.restarts = 150
				self.bh_steps = 50



		# --------------------------------------------- // --------------------------------------------- #
		def get_params(self):
				return self.cv_params

		def set_params(self, params):
				'not done'
				#for keys in params.keys():
				return 0

		def set_acqmax(self, restarts, bh_steps):
				self.restarts = restarts
				self.bh_steps = bh_steps

		# --------------------------------------------- // --------------------------------------------- #
		def cv(self, estimator_params):

				if self.ptypes != None:

						if self.ptypes == 'int':
								for key in estimator_params.keys():
										estimator_params[key] = int(estimator_params[key])

						else:
								for key in self.ptypes.keys():
										estimator_params[key] = self.ptypes[key](estimator_params[key])

				if self.pfixed != None:
						for key in self.pfixed.keys():
								estimator_params[key] = self.pfixed[key]

				if self.plist != None:
						for key in self.plist.keys():
								estimator_params[key] = self.plist[key][int(estimator_params[key]) - 1]


				self.estimator.set_params(**estimator_params)
				v = self.estimator.evaluate(self.cv_params['X'])
				return v

#				self.cv_params['estimator'] = estim

			
#				cvscore = cross_val_score(**self.cv_params)
#				return numpy.mean(cvscore)

		# --------------------------------------------- // --------------------------------------------- #
		def model_opt(self):

				bo = bayes_opt(self.cv, self.pbounds, acq = self.acq, gp_params = self.gp_params,\
											 random_state = self.random_state)

				if self.ptypes != None:
						bo.set_type(self.ptypes)

				if self.plist != None:
						bo.set_list(self.plist)

				max_val, argmax = bo.log_maximize(restarts = self.restarts, bh_steps = self.bh_steps, verbose = 2, num_it = self.niter)

				return argmax


# --------------------------------------------- // --------------------------------------------- #
# --------------------------------------------- // --------------------------------------------- #
		def fit(self, x):

				self.cv_params['X'] = x

				args = self.model_opt()

				return args




################################################################################
##############################____Bayes_Class____###############################
################################################################################

class bayes_opt:
		'''
		________ Bayesian Optimization Class ________

		An object to perform global constrained optimization.


		Parameters
		----------

		f : The function whose maximum is to be found. It must be of the form f(params) where params
				is an 1d-array.
				--- Given a function F(a, b, c, ...) of N variables, a dictionary with the bounds for each variable
				such as {'a' : (0, 1), 'b' : (10, 542), ...} should be passed to the object. ---


		params_dict : The minimum and maximum bounds for the variables of the target function. It has to be a
									dictionary with keys corresponding to the functions arguments for each bound tupple.
									e.g.: {'a' : (0, 1), 'b' : (10, 542), ...}

		kernel : defaults to 'squared_exp', is the kernel to be used in the gaussian process.

		acq : defaults to 'ei' (Expected Improvement), is the acquisition function to be used
					when deciding where to sample next.

		min_log : Parameter dictating whether to find the kernel parameters that lead to the best gp fit
							(maximum likelihood) or to use the specified kernel parameters.


		Member Functions
		----------------

		set_acquisition : Member function to set the acquisition function to be used. Currently implemented
											options are PoI, Probability of Improvement; EI, Expected Improvement; and UCB, upper
											confidence bound, it takes the parameter of the UCB, k, as argument (defaults to 1).

		set_kernel : Member function to set the kernel function to be used. Similar as the for the GP class.

		acq_max : A member function to find the maximum of the acquisition function. It takes a GP object and
							the number os restarts as additional arguments. It uses the scipy object minimize with method 'L-BFGS-B'
							to find the local minima of minus the acquisition function. It reapeats it a number of times to avoid
							falling into local minima.

		maximize : One of the two main methods of this object. It performs bayesian optimization and return the
							 maximum value of the function together with the position of the maximum. A full_output option can be
							 turned on to have the object return all the sampled values of X and Y.

		log_maximize : The other main method of this object, behaves similarly to maximize, however it performs
									 optimization on a log scale of the arguments. This is particularly useful for when the order of
									 magnitude of the maximum bound is much greater than that of the minimum bound. Should be the
									 prefered method for when optimizing the parameters of say, a classifier in the range (0.001, 100),
									 for example.

		initialize : This member function add to the collection of sampled points used by both maximize methods user
								 defined points. It allow the user to have some control over the sampling space, as well as guide the
								 optimizer in the right direction for cases when a number of relevant points are known.
								 A dictionary with values (single or multiple) for each argument should be provided.
								 {'a' : (0, 1, 0.5, 0.4,...), 'b' : (10, 542, 222, 128,...), ...}

		'''

		def __init__(self, f, params_dict, acq = 'ei', gp_params = None, random_state = None):
				'''This is an object to find the global maximum of an unknown function via gaussian processes./n
					 It takes a function of N variables and the lower and upper bounds for each variable as parameters.
					 The function passed to this object should take as array as entry, therefore a function F of N
					 variables should be passed as, f = lambda x: F(x[0],...,x[N-1]).

					 Member variables
					 ----------------

					 ##

					 self.kernel : Stores the kernel of choice as a member variable.

					 self.k_theta : Stores the parameter theta of the kernel.

					 self.k_l : Stores the parameter l of the kernel.

					 self.ac : Stores the acquisition function of choice as a member variable.

					 ##

					 self.keys : Holds the keys of params_dict, which has the variables names in it.

					 self.pdict : Stores params_dict

					 self.bounds : Stores the variables bounds as a numpy array.

					 self.log_bounds : A member variable to store the log scaled bounds, only used if log_minimize is used
														 and the minimum bound is greater than zero for all variables.

					 self.dim : A member variable that stores the dimension of the target function.

					 ##

					 self.user_x : Member variable used to store x values passed to the initialize method.

					 self.user_y : Member variable used to store f(x) values passed to the initialize method.

					 self.user_init : A member variable that keeps track of whether the method 'initialize' has been called.

					 self.min_log : Member variable to store whether maximum likelihood is to be used when fitting the
													gp or not.


				'''

				self.keys = list(params_dict.keys())
				self.pdict = params_dict
				self.dim = len(params_dict)

				self.bounds = []
				for key in params_dict.keys():
						self.bounds.append(params_dict[key])

				self.bounds = numpy.asarray(self.bounds)

				self.log_bounds = 0 * numpy.asarray(self.bounds)

				# ----------------------- // ----------------------- # ----------------------- // ----------------------- #
				for n, pair in enumerate(self.bounds):

						if pair[1] == pair[0]:
								raise RuntimeError('The upper and lower bound of parameter %i are the same, \
								the upper bound must be greater than the lower bound.' % n)

						if pair[1] < pair[0]:
								raise RuntimeError('The upper bound of parameter %i is less than the lower bound, \
								the upper bound must be greater than the lower bound.' % n)


				# ----------------------- // ----------------------- # ----------------------- // ----------------------- #
				self.f = f

				ac = acquisition()
				ac_types = {'ei' : ac.EI, 'poi' : ac.PoI, 'ucb' : ac.UCB}
				try:
						self.ac = ac_types[acq]
				except KeyError:
						print('Custom acquisition function being used.')
						self.ac = acq


				# ----------------------- // ----------------------- # ----------------------- // ----------------------- #
				self.randomstate = random_state
				self.user_x = numpy.empty((1, len(params_dict)))
				self.user_y = numpy.empty(1)


				# ----------------------- // ----------------------- # ----------------------- // ----------------------- #
				# pass parameters to the Gaussian process.
				self.gpparams = gp_params

				# When parameters are floats or elements of a list
				self.ptype = None
				self.plist = None


		# ----------------------- // ----------------------- # ----------------------- // ----------------------- #
		# ----------------------- // ----------------------- # ----------------------- // ----------------------- #
		def set_acquisition(self, acq = 'ucb', k = 1):
				''' Set a new acquisition function.

						Parameters
						----------
						acq : One of the supported acquisition function names or a custom one.

						k : Parameter k of the UCB acquisition function.

						Returns
						-------
						Nothing.
				'''

				ac = acquisition(k)
				ac_types = {'ei' : ac.EI, 'poi' : ac.PoI, 'ucb' : ac.UCB}
				try:
						self.ac = ac_types[acq]
				except KeyError:
						print('Custom acquisition function being used.')
						self.ac = acq

		def set_type(self, types = None):
				self.ptype = types

		def set_list(self, lists = None):
				self.plist = lists

		# ------------------------------ // ------------------------------ # ------------------------------ // ------------------------------ #
		def init(self, init_points, return_log):
				'''A function to perform all initialization and clear the optimize methods - To be constructed'''

				if self.randomstate != None:
						numpy.random.seed(self.randomstate)

				print('Optimization procedure is initializing at %i random points.' % init_points)

				#Sampling some points are random to define xtrain.
				xtrain = numpy.asarray([numpy.random.uniform(x[0], x[1], size = init_points) for x in self.log_bounds]).T
				ytrain = []
				for x in xtrain : 
					ytrain.append(self.f(dict(zip(self.keys, return_log(x)))))
					print('%d points initialized.' % len(ytrain))

				ytrain = numpy.asarray(ytrain)

				print('Optimization procedure is done initializing.')

				return xtrain, ytrain

		# ----------------------- // ----------------------- # ----------------------- // ----------------------- #
		def acq_max(self, gp, ymax, restarts, bh_steps, Bounds):
				''' A function to find the maximum of the acquisition function using the 'L-BFGS-B' method.

						Parameters
						----------
						gp : A gaussian process fitted to the relevant data.

						ymax : The current maximum known value of the target function.

						restarts : The number of times minimation if to be repeated. Larger number of restarts
											 improves the chances of finding the true maxima.

						Bounds : The variables bounds to limit the search of the acq max.


						Returns
						-------
						x_max : The arg max of the acquisition function.
				'''

				x_max = Bounds[:, 0]
				ei_max = 0

				for i in range(restarts):
						#Sample some points at random.
						x_try = numpy.asarray([numpy.random.uniform(x[0], x[1], size = 1) for x in Bounds]).T

						#Find the minimum of minus que acquisition function
						'''
						res = basinhopping(lambda x: -self.ac(x, gp = gp, ymax = ymax), \
															 x0 = x_try, niter=bh_steps, T=2, stepsize=0.1,\
															 minimizer_kwargs = {'bounds' : Bounds, 'method' : 'L-BFGS-B'})
						'''

						res = minimize(lambda x: -self.ac(x, gp = gp, ymax = ymax), x_try, bounds = Bounds, method = 'L-BFGS-B')


						#Store it if better than previous minimum(maximum).
						if -res.fun >= ei_max:
								x_max = res.x
								ei_max = -res.fun

						#print(-res.fun, ei_max)

				return x_max


		# ----------------------- // ----------------------- # ----------------------- // ----------------------- #
		# ----------------------- // ----------------------- # ----------------------- // ----------------------- #
		def log_maximize(self, restarts = 10, bh_steps = 50, num_it = 15, verbose = 2, full_out = False):
				''' Main optimization method perfomed in a log scale.

						Parameters
						----------
						init_points : Number of randomly chosen points to sample the target function before fitting the gp.

						restarts : The number of times minimation if to be repeated. Larger number of restarts
											 improves the chances of finding the true maxima.

						num_it : Total number of times the process is to reapeated. Note that currently this methods does not have
										 stopping criteria (due to a number of reasons), therefore the total number of points to be sampled
										 must be specified.

						verbose : The amount of information to be printed during optimization.
											Accepts 0(nothing), 1(partial), 2(full).

						full_out : If the full output is to be returned or just the function maximum and arg max.


						Returns
						-------
						y_max, x_max : The function maximum and its position.

						y_max, x_max, y, x : In addition to the maximum and arg max, return all the sampled x and y points.

				'''
				total_time = datetime.now()
				init_points = 2 * self.dim + 3

				pi = print_info(verbose, types = self.ptype, lists = self.plist)

				def return_log(x):
						return xmins * (10 ** (x * min_max_ratio))


				# ------------------------------ // ------------------------------ // ------------------------------ #
				for n, pair in enumerate(self.bounds):
						if pair[0] <= 0.0:
								raise RuntimeError('The lower bound of parameter %i is less or equal to zero, \
																		log grid requires strictly positive lower bounds.' % n)

				#Put all the bounds in the 0-1 interval of a log scale.
				self.log_bounds = numpy.log10(self.bounds/self.bounds[:, [0]]) /\
													numpy.log10(self.bounds/self.bounds[:, [0]])[:, [1]]
				min_max_ratio = numpy.log10(self.bounds[:, 1] / self.bounds[:, 0])
				xmins = self.bounds[:, 0]


				# ------------------------------ // ------------------------------ // ------------------------------ #
				xtrain, ytrain = self.init(init_points, return_log)
				ymax = ytrain.max()


				# ------------------------------ // ------------------------------ // ------------------------------ #
				# Fitting the gaussian process
				gp = GaussianProcess()
				if self.gpparams != None:
						gp.set_params(**self.gpparams)

				gp.fit(xtrain, ytrain)

				# Finding argmax of the acquisition function.
				x_max = self.acq_max(gp, ymax, restarts, bh_steps, self.log_bounds)

				for i in range(num_it):
						op_start = datetime.now()

						xtrain = numpy.concatenate((xtrain, x_max.reshape((1, self.dim))), axis = 0)
						ytrain = numpy.append(ytrain, self.f(dict(zip(self.keys, return_log(x_max)))))

						ymax = ytrain.max()

						#Updating the GP.
						gp.fit(xtrain, ytrain)

						# Finding new argmax of the acquisition function.
						x_max = self.acq_max(gp, ymax, restarts, bh_steps, self.log_bounds)
						# Printing everything
						pi.print_log(op_start, i, x_max, xmins, min_max_ratio, ymax, xtrain, ytrain, self.keys)


				tmin, tsec = divmod((datetime.now() - total_time).total_seconds(), 60)
				print('Optimization finished with maximum: %8f | Time taken: %i minutes and %s seconds' % \
							(ytrain.max(), tmin, tsec))

				if full_out:
						return ytrain.max(), dict(zip(self.keys, return_log(xtrain[numpy.argmax(ytrain)]))), \
									 ytrain, return_log(xtrain)
				else:
						return ytrain.max(), dict(zip(self.keys, return_log(xtrain[numpy.argmax(ytrain)])))




################################################################################
################################################################################
################################ Help Functions ################################
################################################################################
################################################################################



################################################################################
############################# Acquisition Functions ############################
################################################################################

class acquisition:
		'''An object to compute the acquisition functions.'''


		def __init__(self, k = 1):
				'''If UCB is to be used, a constant kappa is needed.'''
				self.kappa = k

		def UCB(self, x, gp, ymax):
				mean, var = gp.predict(x, eval_MSE = True)
				return mean + self.kappa * sqrt(var)

		def EI(self, x, gp, ymax):
				mean, var = gp.predict(x, eval_MSE = True)
				if var == 0:
						return 0
				else:
						Z = (mean - ymax)/sqrt(var)
						return (mean - ymax) * norm.cdf(Z) + sqrt(var) * norm.pdf(Z)

		def PoI(self, x, gp, ymax):
				mean, var = gp.predict(x, eval_MSE = True)
				if var == 0:
						return 1
				else:
						Z = (mean - ymax)/sqrt(var)
						return norm.cdf(Z)


################################################################################
################################## Print Info ##################################
################################################################################


class print_info:
		'''A class to take care of the verbosity of the other classes.'''
		'''Under construction!'''

		def __init__(self, level, types = None, lists = None):
				self.lvl = level
				self.timer = 0

				self.ptype = types
				self.plist = lists

		def print_log(self, op_start, i, x_max, xmins, min_max_ratio, ymax, xtrain, ytrain, keys):

				def return_log(x):
						return xmins * (10 ** (x * min_max_ratio))

				parameters_dict = dict(zip(keys, return_log(xtrain[-1])))
				max_dict = dict(zip(keys, return_log(xtrain[numpy.argmax(ytrain)])))

				# ------------------------------ // ------------------------------ #
				if self.ptype != None:
						if self.ptype == 'int':
								for key in parameters_dict.keys():
										parameters_dict[key] = int(parameters_dict[key])
										max_dict[key] = int(max_dict[key])

						else:
								for key in self.ptype.keys():
										parameters_dict[key] = self.ptype[key](parameters_dict[key])
										max_dict[key] = self.ptype[key](max_dict[key])

				if self.plist != None:
						for key in self.plist.keys():
								parameters_dict[key] = self.plist[key][int(parameters_dict[key]) - 1]
								max_dict[key] = self.plist[key][int(max_dict[key]) - 1]
				# ------------------------------ // ------------------------------ #


				if self.lvl == 2:

						numpy.set_printoptions(precision = 4, suppress = True)

						print('Iteration: %3i | Last sampled value: %8f' % ((i+1), ytrain[-1]),\
									'| with parameters: ', parameters_dict)

						print('							 | Current maximum: %11f | with parameters: ' % ymax, \
									max_dict)

						minutes, seconds = divmod((datetime.now() - op_start).total_seconds(), 60)
						print('							 | Time taken: %i minutes and %s seconds' % (minutes, seconds))
						print('')


				elif self.lvl == 1:

						self.timer += (datetime.now() - op_start).total_seconds()

						if (i+1)%10 == 0:
								minutes, seconds = divmod(self.timer, 60)
								print('Iteration: %3i | Current maximum: %f | Time taken: %i minutes and %.2f seconds' % \
											(i+1, ymax, minutes, seconds))
								self.timer = 0

				else:
						pass



if __name__ == '__main__':

		from sklearn.datasets import make_classification
		from sklearn.grid_search import RandomizedSearchCV
		import scipy

		from sklearn.svm import SVC
		from sklearn.ensemble import RandomForestClassifier
		from sklearn.ensemble import GradientBoostingClassifier


		x, y = make_classification(n_samples = 5000, n_features = 75, n_informative = 5,\
															 n_clusters_per_class = 4, random_state = 10)

		bocv = BayesianOptCV(estimator = SVC, param_bounds={'C' : (0.001, 50), 'gamma' : (0.0005, 1)},\
												 #param_list = {'kernel' : ['rbf', 'linear']},\
												 param_fixed = {'random_state' : 1},\
												 n_jobs = 8, cv = 3, n_iter = 50,\
												 gp_params = {'corr' : 'squared_exponential', 'regr' : 'constant'},\
												 acq = 'ei')

		#bocv.fit(x, y)

		randcv = RandomizedSearchCV(estimator = SVC(random_state = 1), \
																param_distributions = {'C' : scipy.stats.uniform(scale = 50),\
																											 'gamma' : scipy.stats.uniform(scale = .1)},\
																verbose = 1,\
																n_iter=50,\
																n_jobs = 8)

		#randcv.fit(x, y)
		#print(randcv.best_score_)
		#print(randcv.best_params_)


		RFcv = BayesianOptCV(estimator = RandomForestClassifier, \
												 param_bounds = {'n_estimators' : (1, 100), 'min_samples_split' : (2, 50)},\
												 param_types = 'int',\
												 param_fixed = {'n_jobs' : -1, 'random_state' : 1},\
												 n_jobs = 3, cv = 3, n_iter = 20,\
												 gp_params = {'corr' : 'absolute_exponential', 'regr' : 'quadratic'},\
												 acq = 'ei')

		#RFcv.fit(x, y)

		pb = {'n_estimators' : (1, 100), 'subsample' : (0.2, 0.99999), 'min_samples_split' : (2, 50),\
					'min_samples_leaf' : (1, 40), 'learning_rate' : (0.01, 0.25), 'max_depth' : (2, 10)}

		pt = {'n_estimators' : int, 'min_samples_split' : int, 'min_samples_leaf' : int, 'max_depth' : int}

		pl = {'max_features' : ['sqrt', 'log2']}

		pf = {'verbose' : 0, 'random_state' : 1}

		GBcv = BayesianOptCV(estimator = GradientBoostingClassifier, \
												 param_bounds = pb,\
												 param_types = pt,\
												 param_list = pl,\
												 param_fixed = pf,\
												 n_jobs = 5, cv = 5, n_iter = 15,\
												 gp_params = {'corr' : 'absolute_exponential', 'regr' : 'constant'},\
												 acq = 'ei')

		GBcv.fit(x, y)

		randcv = RandomizedSearchCV(estimator = GradientBoostingClassifier(random_state = 1), \
																param_distributions = {'n_estimators' : scipy.stats.randint(low = 1, high = 100),\
																											 'subsample' : scipy.stats.uniform(loc = 0.2, scale = 0.799),\
																											 'min_samples_split' : scipy.stats.randint(low = 2, high = 50),\
																											 'min_samples_leaf' : scipy.stats.randint(low = 1, high = 40),\
																											 'learning_rate' : scipy.stats.uniform(loc = 0.01, scale = 0.24),\
																											 'max_depth' : scipy.stats.randint(low = 2, high = 10),\
																											 'max_features' : ['sqrt', 'log2']},\
																verbose = 1,\
																cv = 5,
																n_iter=32,\
																n_jobs = 8)

		randcv.fit(x, y)
		print(randcv.best_score_)
		print(randcv.best_params_)
