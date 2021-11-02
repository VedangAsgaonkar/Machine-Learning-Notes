It is a convention in most ML libraries to represent time series data as N x T x D,
where T is the number of timesteps over which the data was measures, D is the number
of features that were measured in every timestep and N is the number of data samples (as 
all the samples are stacked together for vectorization)
