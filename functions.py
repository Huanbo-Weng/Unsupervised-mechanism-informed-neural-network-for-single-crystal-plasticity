import numpy as np
import torch
import math

class dataset:
    def __init__(self, Model):
        self.Model = Model
        path = 'dataset/' + Model + '/data.csv'
        self.data = np.loadtxt(path, delimiter=',')
        path = 'dataset/' + Model + '/label.csv'
        self.label = np.loadtxt(path, delimiter=',')
        # self.dgamma = np.loadtxt(path, delimiter=',') 
        self.dataNum = self.data.shape[0]
        np.random.seed(1)
        # self.dataSep()
    
    def transToSlip(self, args):
        Dglobal, slipDef, Ddemsd = D_slip(args)
        tau = np.dot(self.data[:,:6],slipDef.T)
        self.nnData = tau
        if 'phi_n' in args.FTcomp:
            slipNomal = octPlaneNormal(args)   # Resolved along slip plane normal direction
            fabricTensor = np.zeros([self.data.shape[0], 6])
            fabricTensor[:,0] = self.data[:,19] + self.data[:,20]
            fabricTensor[:,1] = self.data[:,19] - self.data[:,20]
            fabricTensor[:,3] = self.data[:,21]
            slip_fT = np.dot(fabricTensor,slipNomal.T)
            self.nnData = np.concatenate((self.nnData, slip_fT), axis=1)            
        if 'phi_s' in args.FTcomp:
            fabricTensor = np.zeros([self.data.shape[0], 6])
            fabricTensor[:,0] = self.data[:,19] + self.data[:,20]
            fabricTensor[:,1] = self.data[:,19] - self.data[:,20]
            fabricTensor[:,3] = self.data[:,21]
            slip_fT = np.dot(fabricTensor,slipDef.T)
            self.nnData = np.concatenate((self.nnData, slip_fT), axis=1)
        if 'phi_s_iso' in args.FTcomp:
            fabricTensor_iso = np.zeros([self.data.shape[0], 6])
            fabricTensor_iso[:,0] = self.data[:,19]
            fabricTensor_iso[:,1] = self.data[:,19]
            slip_fT_iso = np.dot(fabricTensor_iso,slipDef.T)
            self.nnData = np.concatenate((self.nnData, slip_fT_iso), axis=1)
        if 'phi_s_dev' in args.FTcomp: 
            fabricTensor_dev = np.zeros([self.data.shape[0], 6])
            fabricTensor_dev[:,0] = self.data[:,20]
            fabricTensor_dev[:,1] = -self.data[:,20]
            fabricTensor_dev[:,3] = self.data[:,21]
            slip_fT_dev = np.dot(fabricTensor_dev,slipDef.T)
            self.nnData = np.concatenate((self.nnData, slip_fT_dev), axis=1)              
        self.nnData = np.concatenate((self.nnData,self.data[:,:19]), axis=1)
        return self.nnData
    
    def dataSep(self):
        dataSize = self.dataNum//10
        # create validation set
        index = np.random.choice(self.data.shape[0], size=dataSize, replace=False, p=None)
        self.x_validation = self.data[index]
        self.y_validation = self.label[index]  
        # self.dgamma_validation = self.dgamma[index]          
        self.data = np.delete(self.data,index,0)
        self.label = np.delete(self.label,index,0)
        # self.dgamma = np.delete(self.dgamma,index,0)
        # create test set
        index = np.random.choice(self.data.shape[0], size=dataSize, replace=False, p=None)
        self.x_test = self.data[index]
        self.y_test = self.label[index]
        # self.dgamma_test = self.dgamma[index]
        # obtain training set  
        self.x_train = np.delete(self.data,index,0)
        self.y_train = np.delete(self.label,index,0)
        # self.dgamma_train = np.delete(self.dgamma,index,0)
        
        print('Shape of training dataset: data ' + str(self.x_train.shape) + \
              ' ,label ' + str(self.y_train.shape))
        print('Shape of validation dataset: data ' + str(self.x_validation.shape) + \
              ' ,label ' + str(self.y_validation.shape))
        print('Shape of test dataset: data ' + str(self.x_test.shape) + \
              ' ,label ' + str(self.y_test.shape))                


'''======= Functions for crystal plasticity ======='''
def sind(x):
    return math.sin(x * math.pi / 180)

def cosd(x):
    return math.cos(x * math.pi / 180)

def unitVec(vector):   
    norm = np.linalg.norm(vector)
    norm_vector = vector / norm
    return norm_vector

def deleteNeg(vector):   
    vector[vector<0] = 0
    return vector

def Dmatrix(args):
    ## Cubic material
    Dlocal = np.zeros([6,6])
    for i in range(3):
        Dlocal[i,i] = args.C1111 
        for j in range(3):
            if i != j:
                Dlocal[i,j] = args.C1122
        Dlocal[i+3,i+3] = args.C1212
    return Dlocal

def rotation(args):
    a_local = unitVec(args.vecLocal1)
    c_local = unitVec(args.vecLocal2)
    b_local = np.cross(a_local,c_local)
    coord_local = np.array([a_local,b_local,c_local]).T
    
    a_global = np.array([0,0,1])
    c_global = np.array([1,0,0])
    b_global = np.cross(a_global,c_global)
    coord_global = np.array([a_global,b_global,c_global]).T
    
    coord_local_inv = np.linalg.inv(coord_local)
    rotate = np.dot(coord_global,coord_local_inv).T
    return rotate


def rotMat_D(rotate):
    rotateD = np.zeros([6,6])
    for j in range(3):
        j1 = j//2
        j2 = 1+math.ceil(j/3)
        for i in range(3):
            i1 = i//2
            i2 = 1+math.ceil(i/3)
            rotateD[i,j] = rotate[i,j]**2
            rotateD[i,j+3] = rotate[i,j1]*rotate[i,j2]
            rotateD[i+3,j] = 2*rotate[i1,j]*rotate[i2,j]
            rotateD[i+3,j+3] = rotate[i1,j1]*rotate[i2,j2] + \
                               rotate[i1,j2]*rotate[i2,j1]
    return rotateD

def octPlaneNormal(args):
    iwkNor = np.zeros([args.slipSys['oct'],3])
    iwkDir = np.zeros([args.slipSys['oct'],3])
    
    slipNor = unitVec([1,1,1])
    iwkNor[0:3,:] = slipNor
    for i in range(3):
        temNorm = slipNor.copy()
        temNorm[i] = -temNorm[i]
        iwkNor[3*(i+1):3*(i+2),:] = temNorm
    
    slipNor = unitVec([1,1,1])
    iwkDir[0:3,:] = slipNor
    for i in range(3):
        temNorm = slipNor.copy()
        temNorm[i] = -temNorm[i]
        iwkDir[3*(i+1):3*(i+2),:] = temNorm
    
    slipNor = iwkNor
    slipDir = iwkDir
    slipDef = np.zeros([args.slipSys['oct'],6])
    for i in range(args.slipSys['oct']):
        slipDef[i,0] = slipNor[i,0]*slipDir[i,0]
        slipDef[i,1] = slipNor[i,1]*slipDir[i,1]
        slipDef[i,2] = slipNor[i,2]*slipDir[i,2]
        slipDef[i,3] = slipNor[i,0]*slipDir[i,1] + slipNor[i,1]*slipDir[i,0]
        slipDef[i,4] = slipNor[i,0]*slipDir[i,2] + slipNor[i,2]*slipDir[i,0]
        slipDef[i,5] = slipNor[i,1]*slipDir[i,2] + slipNor[i,2]*slipDir[i,1]    
    return slipDef

def octSlipSys(args):
    iwkNor = np.zeros([args.slipSys['oct'],3])
    iwkDir = np.zeros([args.slipSys['oct'],3])
    
    slipNor = unitVec([1,1,1])
    iwkNor[0:3,:] = slipNor
    for i in range(3):
        temNorm = slipNor.copy()
        temNorm[i] = -temNorm[i]
        iwkNor[3*(i+1):3*(i+2),:] = temNorm
    
    iwkDir[0,:] = unitVec([0,-1,1])
    iwkDir[1,:] = unitVec([1,0,-1])
    iwkDir[2,:] = unitVec([-1,1,0])
    iwkDir[3,:] = unitVec([1,0,1])
    iwkDir[4,:] = unitVec([1,1,0])
    iwkDir[5,:] = unitVec([0,-1,1])
    iwkDir[6,:] = unitVec([0,1,1])
    iwkDir[7,:] = unitVec([1,1,0])
    iwkDir[8,:] = unitVec([1,0,-1])
    iwkDir[9,:] = unitVec([0,1,1])
    iwkDir[10,:] = unitVec([1,0,1])
    iwkDir[11,:] = unitVec([-1,1,0])
    return iwkNor,iwkDir

def cubSlipSys(args):
    iwkNor = np.zeros([args.slipSys['cub'],3])
    iwkDir = np.zeros([args.slipSys['cub'],3])
    
    iwkNor[0,:] = unitVec([1,0,0])
    iwkNor[1,:] = unitVec([1,0,0])
    iwkNor[2,:] = unitVec([0,1,0])
    iwkNor[3,:] = unitVec([0,1,0])
    iwkNor[4,:] = unitVec([0,0,1])
    iwkNor[5,:] = unitVec([0,0,1])
    
    iwkDir[0,:] = unitVec([0,1,1])
    iwkDir[1,:] = unitVec([0,-1,1])
    iwkDir[2,:] = unitVec([1,0,1])
    iwkDir[3,:] = unitVec([1,0,-1])
    iwkDir[4,:] = unitVec([1,1,0])
    iwkDir[5,:] = unitVec([-1,1,0])
    return iwkNor,iwkDir

def D_slip(args):
    Dlocal = Dmatrix(args)
    rotate = rotation(args)
    rotateD = rotMat_D(rotate)
    '''======= Rotation matrix of Dlocal ======='''
    Dglobal = np.dot(rotateD.T,Dlocal)
    Dglobal = np.dot(Dglobal,rotateD)
    '''========== Calculate schmid factor ==========='''
    # nslipSum = sum(args.nslip.values())
    slipDefAssembly = np.empty([0,6])
    for key in args.slipSys.keys():
        if key == 'oct':
            slipNor, slipDir = octSlipSys(args)
        elif key == 'cub':
            slipNor, slipDir = cubSlipSys(args)
            
        slipDef = np.zeros([args.slipSys[key],6])
        for i in range(args.slipSys[key]):
            slipDef[i,0] = slipNor[i,0]*slipDir[i,0]
            slipDef[i,1] = slipNor[i,1]*slipDir[i,1]
            slipDef[i,2] = slipNor[i,2]*slipDir[i,2]
            slipDef[i,3] = slipNor[i,0]*slipDir[i,1] + slipNor[i,1]*slipDir[i,0]
            slipDef[i,4] = slipNor[i,0]*slipDir[i,2] + slipNor[i,2]*slipDir[i,0]
            slipDef[i,5] = slipNor[i,1]*slipDir[i,2] + slipNor[i,2]*slipDir[i,1]
        slipDefAssembly = np.concatenate((slipDefAssembly, slipDef), axis=0)
    
    Ddemsd = np.dot(slipDefAssembly,Dglobal)  
    return Dglobal, slipDefAssembly, Ddemsd

def rotateToPrinStress(stress,theta,phi):
    rotz = np.array([[cosd(theta),-sind(theta),0],[sind(theta),cosd(theta),0],[0,0,1]])
    rotx = np.array([[1,0,0],[0,cosd(phi),-sind(phi)],[0,sind(phi),cosd(phi)]])
    for i in range(stress.shape[0]):
        stressTensor = np.array([[stress[i,0],stress[i,3],stress[i,4]],
                                  [stress[i,3],stress[i,1],stress[i,5]],
                                [stress[i,4],stress[i,5],stress[i,2]]])
        R = np.dot(rotz,rotx)
        stressT = np.dot(np.dot(R.T,stressTensor),R)
        stress[i,0] = stressT[0,0]
        stress[i,1] = stressT[1,1]
        stress[i,2] = stressT[2,2]
        stress[i,3] = stressT[0,1]
        stress[i,4] = stressT[0,2]
        stress[i,5] = stressT[1,2]
    return stress
                            
def rotateToPrinStrain(stress,theta,phi):
    rotz = np.array([[cosd(theta),-sind(theta),0],[sind(theta),cosd(theta),0],[0,0,1]])
    rotx = np.array([[1,0,0],[0,cosd(phi),-sind(phi)],[0,sind(phi),cosd(phi)]])
    for i in range(stress.shape[0]):
        stressTensor = np.array([[stress[i,0],stress[i,3]/2,stress[i,4]/2],
                                 [stress[i,3]/2,stress[i,1],stress[i,5]/2],
                                [stress[i,4]/2,stress[i,5]/2,stress[i,2]]])
        R = np.dot(rotz,rotx)
        stressT = np.dot(np.dot(R.T,stressTensor),R)
        stress[i,0] = stressT[0,0]
        stress[i,1] = stressT[1,1]
        stress[i,2] = stressT[2,2]
        stress[i,3] = stressT[0,1]
        stress[i,4] = stressT[0,2]
        stress[i,5] = stressT[1,2]
    return stress    


'''======= Functions for neural network ======='''
def reflectInput(data,stateID):
    size = len(stateID)
    normalizedData = data.clone()
    maxX = torch.zeros([size]) 
    for i in range(size):
        maxX[i] = torch.max(torch.max(data[:,stateID[i]]), axis=0)[0]
    minX = torch.zeros([size])
    for i in range(size):
        minX[i] = torch.min(torch.min(data[:,stateID[i]]), axis=0)[0]    
    for i in range(size):
        normalizedData[:,stateID[i]] =  (data[:,stateID[i]] - minX[i]) / (maxX[i] - minX[i])     
    normalizedData = torch.where(torch.isnan(normalizedData), \
                                 torch.ones_like(normalizedData), normalizedData)                             
    return normalizedData,maxX,minX

def reflectOutput(data):
    maxX = torch.max(torch.max(data), axis=0)[0]
    minX = torch.min(torch.min(data), axis=0)[0] 
    normalizedData = (data - minX) / (maxX - minX)
    normalizedData = torch.where(torch.isnan(normalizedData), \
                             torch.ones_like(normalizedData), normalizedData)  
    return normalizedData,maxX,minX

def testReflectInput(data,maxX,minX,stateID):
    size = len(stateID)
    normalizedData = data.clone()
    for i in range(size):
        normalizedData[:,stateID[i]] =  (data[:,stateID[i]] - minX[i]) / (maxX[i] - minX[i])                                  
    normalizedData = torch.where(torch.isnan(normalizedData), \
                                 torch.ones_like(normalizedData), normalizedData)      
    return normalizedData

def testReflectOutput(data,maxX,minX):
    normalizedData = (data - minX) / (maxX - minX)
    return normalizedData

def reflectBackInput(data,maxX,minX,stateID):
    size = len(stateID)
    trueData = data.clone()
    for i in range(size):
        trueData[:,stateID[i]] =  data[:,stateID[i]] * (maxX[i] - minX[i]) +  minX[i]                                
    return trueData

def reflectBackOutput(data,maxX,minX):
    trueData = data * (maxX - minX) +  minX
    return trueData


    
    
    
    

