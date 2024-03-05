class Modified_Hopfield(object):
    
    def __init__(self):
        pass
        
    def get_attractors_from_data(self,window_FR,trial_type_encoding):
        # Needs pre-processed experimant data: 
        # normalized log neural firing rates of each win-dows saved as window_FR, and behavior output saved as trial_type_encoding
        # Calculate the mean firing rates of each kind of behavior outputs
        mean_1 = np.zeros([np.shape(window_FR)[1]])
        mean_2 = np.zeros([np.shape(window_FR)[1]])
        mean_4 = np.zeros([np.shape(window_FR)[1]])
        for i in range(len(trial_type_encoding)):
            if trial_type_encoding[i] == 1:
                mean_1 += window_FR[i,:]
            elif trial_type_encoding[i] == 2:
                mean_2 += window_FR[i,:]
            elif trial_type_encoding[i] == 4:
                mean_4 += window_FR[i,:]
        mean_1 = mean_1/np.sum(np.array(trial_type_encoding) == 1)
        mean_2 = mean_2/np.sum(np.array(trial_type_encoding) == 2)
        mean_4 = mean_4/np.sum(np.array(trial_type_encoding) == 4)
        # mean log normalized firing rates are trans-ferred to -1 and 1 for construction of attractor states
        sgn_mean1 = np.sign(mean_1)
        sgn_mean2 = np.sign(mean_2)
        sgn_mean4 = np.sign(mean_4)
        # only neurons with opposite signs of mean nor-malized log firing rate for "food" and "water" are chosen
        # this process makes chosen_sgn_mean2 = - cho-sen_sgn_mean1, so food and water needs can be modeled by one pair of attractors
        chosen_sgn_mean1 = sgn_mean1[sgn_mean1*sgn_mean2 == -1]
        chosen_sgn_mean4 = sgn_mean4[sgn_mean1*sgn_mean2 == -1]
        N = len(chosen_sgn_mean1)
        # construct the connection matrix M using 2 at-tractor states template1 (food and water needs) and tem-plate4 (other needs)
        s1 = (np.reshape(chosen_sgn_mean1,[-1,1]))*np.transpose(np.reshape(chosen_sgn_mean1,[-1,1]))
        s2 = (np.reshape(chosen_sgn_mean4,[-1,1]))*np.transpose(np.reshape(chosen_sgn_mean4,[-1,1]))
        M = s1+s2
        M = M/(2*N)
        self.N = N
        self.M = M
        self.template1 = chosen_sgn_mean1
        self.template4 = chosen_sgn_mean4
        self.chosen_label = sgn_mean1*sgn_mean2 == -1
        
    def activation_func(self,input_x):
        # activation functions depend on template1, run-ning_k1 (hungry level) and running_k2 (thirsty level)
        sgn = np.maximum(0,self.running_k1*input_x*self.template1)*self.template1+np.minimum(0,self.running_k2*input_x*self.template1)*self.template1
        sgn = np.minimum(1,sgn)
        sgn = np.maximum(-1,sgn)
        return sgn
    
    def run_model(self,initial_x,n_trial,nl,lr,sr,k1,k2):
        self.n_trial = n_trial # number of trials
        self.nl = nl # noise level
        self.lr = lr # updating rate
        self.sr = sr # satiation rate
        self.initial_k1 = k1 # initial hungry level
        self.initial_k2 = k2 # initial thirsty level
        self.running_k1 = k1 # current hungry level, sa-tiated a little bit after food choice
        self.running_k2 = k2 # current thirsty level, sa-tiated a little bit after water choice
        xs = np.zeros((self.N,n_trial))
        xs[:,0] = initial_x
        # project the neural pattern onto thirst&hunger dimension (pros) and other needs dimension (pros4)
        pros = np.zeros(n_trial)
        pros[0] = np.dot(xs[:,0],self.template1)/np.dot(self.template1,self.template1)
        pros4 = np.zeros(n_trial)
        pros4[0] = np.dot(xs[:,0],self.template4)/np.dot(self.template4,self.template4)
        # determine the output of initial trial
        output = np.zeros(n_trial)
        if np.abs(pros4[0])>np.abs(pros[0]):
            output[0] = 3
        elif pros[0] < 0:
            output[0] = 2
            if self.running_k2>0:
                self.running_k2 -= sr
        else:
            output[0] = 1
            if self.running_k1>0:
                self.running_k1 -= sr
        # run trials
        for i in range(1,n_trial):
            h = np.random.randn(self.N)
            xs[:,i] = xs[:,i-1] + h*nl + lr*(-xs[:,i-1] + self.activation_func(np.reshape(self.M@np.reshape(xs[:,i-1],[-1,1]),[-1])))
            pros[i] = np.dot(xs[:,i],self.template1)/np.dot(self.template1,self.template1)
            pros4[i] = np.dot(xs[:,i],self.template4)/np.dot(self.template4,self.template4)
            # output of each trial determined by the pro-jections plus a noise, modelling the neural noise after the onset of odor until the choice
            pros_rand = pros[i] + 0.2*np.random.randn()
            pros4_rand = pros4[i] + 0.2*np.random.randn()
            if np.abs(pros4_rand)>np.abs(pros_rand):
                output[i] = 3
            elif pros_rand < 0:
                output[i] = 2
                if self.running_k2>0:
                    self.running_k2 -= sr
            else:
                output[i] = 1
                if self.running_k1>0:
                    self.running_k1 -= sr
        # calculate the transition matrix
        trans_matrix = np.zeros((3,3))
        for i in range(1,len(output)):
            trans_matrix[int(output[i-1]-1),int(output[i]-1)] += 1
        self.trans_series = output
        self.trans_matrix = trans_matrix
        self.xs = xs
        self.pros = pros
        self.pros4 = pros4 
