import matplotlib.pyplot as plt
import numpy as np
import h5py
from functions import *
# from PINN_CP import preTrain, train, test
from MINN_RVE import *
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
        self.r0 = 140 # MPa
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
        self.RESUME_FILE = '1ele'
        self.CHECKPOINT_FILE = 'try2'

args = params()

# Specify dataset
Model = 'tension_1ele'
data = dataset(Model)
x = data.transToSlip(args)
# # y = data.label

tau = x[:,args.stateID[0]]
slip_fT_iso = x[:,12:24]
slip_fT_dev = x[:,24:36]
delta_t = x[:,args.stateID[1]]

# # Initialize parameters
# Dglobal, slipDef, Ddemsd = D_slip(args)
# tau = np.dot(x[:,:6],slipDef.T)

  
''' ============= Post analysis ============ '''
outputs, stress_upd = test_StrainRate(data, args)
gradient = test_gradient(data, args)

gamma_Rate = (outputs/delta_t)
micro = 3
gradient = gradient[99*micro:99*(micro+1),:]
tau = tau[99*micro:99*(micro+1),:]
gamma_Rate = gamma_Rate[99*micro:99*(micro+1),:]
slip_fT_iso = slip_fT_iso[99*micro:99*(micro+1),:]
slip_fT_dev = slip_fT_dev[99*micro:99*(micro+1),:]

# curve = 0
# tau = tau[33*curve:33*(curve+1),:]
# gamma_Rate = gamma_Rate[33*curve:33*(curve+1),:]
# slip_fT_iso = slip_fT_iso[33*curve:33*(curve+1),:]
# slip_fT_dev = slip_fT_dev[33*curve:33*(curve+1),:]


# final_gR = gamma_Rate[41::43*3,:]
# unislipFT_iso = slip_fT_iso[41::43*3,:]
# unislipFT_dev = slip_fT_dev[41::43*3,:]


''' ============= Plot figure ============ '''
plt.figure(2, figsize=(8, 6))  # 设置图形大小

plot_x = tau.flatten()
plot_y = gamma_Rate.flatten()

# 绘制散点图
plt.scatter(plot_x, plot_y, alpha=0.6)  # 添加透明度

# 添加标题和坐标轴标签
# plt.title('Relationship between Shear Stress and Shear Strain Rate', fontsize=16)
plt.xlabel('Shear Stress (MPa)', fontsize=14)
plt.ylabel('Shear Strain Rate', fontsize=14)

# 添加网格线
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# 设置刻度标签的大小
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# 显示图形
plt.show()












    