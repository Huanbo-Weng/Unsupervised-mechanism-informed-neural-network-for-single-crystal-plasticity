import matplotlib.pyplot as plt
import numpy as np
from functions import *
from MINN_RVE import *
from ICMINN import *
from postEval import *
import time

class params:
    def __init__(self):
        # elastic parameters
        self.C1111 = 169727 # MPa
        self.C1122 = 104026 # MPa
        self.C1212 = 86000 # MPa
        # define the vector in local coordinate
        self.vecLocal1 = np.array([0,0,1]) # global [001] direction
        self.vecLocal2 = np.array([1,0,0]) # global [100] direction
        # crystal plasticity parameters
        self.slipSys = {'oct':12} # information of slip systems
        self.nslip = sum(self.slipSys.values())
        self.dgamma0 = 0.006056744
        self.r0 = 0 # MPa
        self.g = 322.5 # MPa
        self.n = 3.9 
        self.c1 = 100 # MPa
        self.c2 = 2
        self.b = 42
        self.Q = 1.5 # MPa
        self.H = np.ones([self.nslip,self.nslip])
        self.theta = 0.5
        # hyperparameters for training
        self.learning_Rate = 0.01    
        self.lambda_data = 0
        self.lambda_phy = 1
        self.EPOCHS = 1000
        self.BATCH_SIZE = 128
        self.NORM = 1
        self.seed = 10
        self.FTcomp = []
        self.net = 'NN_strainRate_1ele'
        # data 
        if self.net == 'NN_strainRate_1ele' or 'ICNN_strainRate_1ele':
            self.stateID = [[i for i in range(0,12)],[24],
                            [i for i in range(12,18)],[i for i in range(25,31)]] 
        if len(self.FTcomp) == 1:
            self.stateID = [[i for i in range(0,12)],[36],
                            [i for i in range(24,30)],[i for i in range(37,43)]]
        elif len(self.FTcomp) == 2:
            self.stateID = [[i for i in range(0,12)],[48],
                            [i for i in range(36,42)],[i for i in range(49,55)]]        
        self.RESUME_FILE = 'try'
        self.CHECKPOINT_FILE = 'try'

# Initialize parameters
args = params()
Dglobal, slipDef, Ddemsd = D_slip(args)

# Specify dataset 
Model = 'tension_1ele'
data = dataset(Model)
x = data.transToSlip(args)
y = data.label

''' Training '''
# loss_record, valLoss_record = preTrain(data, args, RESUME=0)     # pure data-driven
loss_record = MINN_train(data, args, RESUME=0)                   # unsupervised MINN
# loss_record = MINN_train_gradient(data, args, RESUME=0)          # unsupervised MINN with gradient as a soft constraint
# loss_record = ICMINN_train(data, args, RESUME=0)                 # unsupervised MINN with gradient as a hard constraint

''' Cyclic Training '''
# for case in range(1,10):
#     args.seed = case
#     args.RESUME_FILE = 'tryGrad2_seed{}'.format(case)
#     args.CHECKPOINT_FILE = 'tryGrad2_seed{}'.format(case)
#     loss_record = MINN_train_gradient(data, args, RESUME=0)
#     outputs, stress_upd = test_StrainRate(data, args)


''' Test '''
outputs, stress_upd = test_StrainRate(data, args)
error = stress_upd-y
stress_upd = stress_upd.flatten()
# y = y.flatten()




    