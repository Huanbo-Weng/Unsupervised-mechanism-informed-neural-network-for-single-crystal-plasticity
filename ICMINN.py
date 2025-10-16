import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import h5py
from torch.optim import lr_scheduler
from torch.utils.data.dataset import Dataset
from functions import *
import time

## create model
import torch
import torch.nn as nn

class ICNN_strainRate_1ele(nn.Module):
    def __init__(self):
        super(ICNN_strainRate_1ele, self).__init__()
        # Input dimensions:
        # - y: 1D (slip variable, convex input 1)
        # - z: 1D (delta t, convex input 2)
        # Network output is convex w.r.t. both y and z.
        
        # --------------------------
        # ICNN layers: all weights must be non-negative (critical for convexity in both variables)
        # --------------------------
        # 1st layer: processes combined features of y and z (weights ≥0)
        self.layer1 = nn.Linear(2, 32)  # Input: [y, z] (2D) → 32D features
        
        # 2nd layer: processes features from layer1 (weights ≥0)
        self.layer2 = nn.Linear(32, 32)  # 32D → 32D
        
        # 3rd layer: processes features from layer2 (weights ≥0)
        self.layer3 = nn.Linear(32, 32)  # 32D → 32D
        
        # Output layer: final mapping (weights ≥0)
        self.out = nn.Linear(32, 1)  # 32D → 1D output
        
        # Initialize all weights to be non-negative
        self.layer1.weight.data.clamp_(min=0.0)
        self.layer2.weight.data.clamp_(min=0.0)
        self.layer3.weight.data.clamp_(min=0.0)
        self.out.weight.data.clamp_(min=0.0)

    def forward(self, x):
        # Split input into two convex variables: y (slip) and z (delta t)
        slip_vars = x[:, :12]  # First 12 features: slip variables (y groups)
        grouped_y = [slip_vars[:, i::12] for i in range(12)]  # 12 groups of y (each 1D)
        z = x[:, 24:25]  # z: delta t (1D, shared across all groups)

        outputs = []
        for y in grouped_y:
            # Combine y and z into a 2D input (both need convexity)
            input_combined = torch.cat([y, z], dim=1)  # Shape: (batch_size, 2)
            
            # --------------------------
            # Forward pass (convex w.r.t. both y and z)
            # --------------------------
            h1 = self.layer1(input_combined)
            h1 = torch.relu(h1)  # Non-decreasing convex activation
            
            h2 = self.layer2(h1)
            h2 = torch.relu(h2)
            
            h3 = self.layer3(h2)
            h3 = torch.relu(h3)
            
            output = self.out(h3)
            outputs.append(output)

        # Concatenate 12 outputs
        final_output = torch.cat(outputs, dim=1)
        return final_output
    
    def apply_constraints(self):
        # Enforce non-negativity on all weights (critical for convexity in both variables)
        with torch.no_grad():
            self.layer1.weight.clamp_(min=0.0)
            self.layer2.weight.clamp_(min=0.0)
            self.layer3.weight.clamp_(min=0.0)
            self.out.weight.clamp_(min=0.0)

class NN_strainRate(nn.Module):
    def __init__(self):
        super(NN_strainRate, self).__init__()
        # Define the neural network
        self.regress = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Extract the first 24 elements at intervals of 12
        slip_vars = x[:, :24]
        grouped_vars = [slip_vars[:, i::12] for i in range(12)]
        other_vars = x[:, 36:37]  # delta t

        outputs = []
        for group in grouped_vars:
            # Combine each group of data with data from 36:37 to form 3 elements
            combined = torch.cat((group, other_vars), dim=1)
            # Input to the neural network to get 1 output
            output = self.regress(combined)
            outputs.append(output)

        # Combine 12 outputs
        final_output = torch.cat(outputs, dim=1)
        return final_output

class NN_strainRate_decoupled(nn.Module):
    def __init__(self):
        super(NN_strainRate_decoupled, self).__init__()
        # Define the neural network
        self.regress = nn.Sequential(
            nn.Linear(4, 32),
            nn.LeakyReLU(negative_slope=0.05, inplace=True), 
            nn.Linear(32, 32),           
            nn.LeakyReLU(negative_slope=0.05, inplace=True), 
            nn.Linear(32, 32),                  
            nn.LeakyReLU(negative_slope=0.05, inplace=True), 
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Extract the first 36 elements at intervals of 12
        slip_vars = x[:, :36]
        grouped_vars = [slip_vars[:, i::12] for i in range(12)]
        other_vars = x[:, 48:49]  # delta t

        outputs = []
        for group in grouped_vars:
            # Combine each group of data with data from 48:49 to form 4 elements
            combined = torch.cat((group, other_vars), dim=1)
            # Input to the neural network to get 1 output
            output = self.regress(combined)
            outputs.append(output)

        # Combine 12 outputs
        NN_output = torch.cat(outputs, dim=1)
        return NN_output


## create subDataset
class subDataset(Dataset):
    # Initialization: define data content and labels
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    # Return the size of the dataset
    def __len__(self):
        return len(self.Data)
    # Get data content and labels
    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label


def preTrain(data, args, RESUME=0):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    
    ## set hyper-parameter
    learning_Rate = args.learning_Rate
    EPOCHS = args.EPOCHS
    BATCH_SIZE = args.BATCH_SIZE
    CHECKPOINT_FILE = 'saved_model/' + args.CHECKPOINT_FILE
    
    '''======= Initialize crystal plasticity parameters ========'''
    dgamma0 = args.dgamma0
    g = args.g
    n = args.n
    c1 = args.c1
    c2 = args.c2
    b = args.b
    Q = args.Q
    H = torch.Tensor(args.H)
    theta = args.theta  
    Dglobal, slipDef, Ddemsd = D_slip(args)
    
    '''========== Start PyTorch ==========='''
    Dglobal_torch = torch.FloatTensor(Dglobal)
    slipDef_torch = torch.FloatTensor(slipDef)
    Ddemsd_torch = torch.FloatTensor(Ddemsd)
    x_train = torch.FloatTensor(data.x_train).to(device)
    y_train = torch.FloatTensor(data.y_train).to(device)
    x_validation = torch.FloatTensor(data.x_validation).to(device)
    y_validation = torch.FloatTensor(data.y_validation).to(device)
        
    if args.NORM == 1:
        x_train,maxx,minx = reflectInput(x_train,args.stateID) 
        y_train,maxy,miny = reflectOutput(y_train)
        x_validation = testReflectInput(x_validation,maxx,minx,args.stateID)
        y_validation = testReflectOutput(y_validation,maxy,miny)
    
    ## set mini_batches
    torch_dataset = subDataset(x_train[:,:], y_train)
    loader = Data.DataLoader(dataset=torch_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=0,
                              )

    if args.net == 'ANN':
        ann = ANN().to(device)
    elif args.net == 'ANNsepInput':
        ann = ANNsepInput().to(device)
    elif args.net == 'ANNsepOutput':
        ann = ANNsepOutput().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ann.parameters(), lr=learning_Rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    print(ann)
    
    loss_record = []
    valLoss_record = []
    best_record = 10000
    min_Epoch = 50
    Flag = 0
    
    if RESUME:
        print("Resume from checkpoint...")
        checkpoint = torch.load(CHECKPOINT_FILE,map_location=torch.device('cpu'))     ## read previous state
        ann.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initepoch = checkpoint['epoch']+1
    else:
        initepoch = 0
     
    start_time = time.time()
    
    for epoch in range(initepoch, EPOCHS):
        for i, (data, labels) in enumerate(loader):

            if torch.cuda.is_available():  
                data = data.cuda()
                labels = labels.cuda()
            
            outputs = ann(data) 
           
            loss = criterion(outputs,labels)
            optimizer.zero_grad()
            loss.backward()  # back pop
            optimizer.step()
            
            if epoch == 350:
                print(1)
            
            if (i + 1) % 1 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.8f'
                      % (epoch + 1, EPOCHS, i + 1, len(x_train) // BATCH_SIZE, loss.item()))
   
        scheduler.step()
        val_out = ann(x_validation)
        val_loss = criterion(val_out, y_validation)
        
        if epoch > min_Epoch and best_record > val_loss.item():

            # torch.save(checkpoint, CHECKPOINT_FILE+'{}'.format(epoch))
            best_record = val_loss.item()
            checkpoint = {'epoch': epoch+1,
                'model_state_dict': ann.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':loss_record
                }
            torch.save(checkpoint, CHECKPOINT_FILE)             
            # !save the last model or the best model
        if loss.item() < 1000:
            Flag = 1
        if Flag == 1:
            loss_record.append(loss.item())
            valLoss_record.append(val_loss.item())
    
    # print('mean_train = ' + str(sum(loss_record[-100:]) / len(loss_record[-100:]))) 
    # print('mean_test = ' + str(sum(valLoss_record[-100:]) / len(valLoss_record[-100:]))) 
    
    end_time = time.time()
    print('time:' + str(end_time - start_time))
    
    plt.plot(loss_record)
    plt.plot(valLoss_record)
    plt.legend(['Training_set','Validation set'])
    plt.ylabel('loss_value')
    plt.xlabel('epochs')
    # plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return loss_record,valLoss_record


def train(data, args, RESUME=0):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    
    ## set hyper-parameter
    learning_Rate = args.learning_Rate
    EPOCHS = args.EPOCHS
    BATCH_SIZE = args.BATCH_SIZE
    CHECKPOINT_FILE = 'saved_model/' + args.CHECKPOINT_FILE
    RESUME_FILE = 'saved_model/' + args.RESUME_FILE
    
    '''======= Initialize crystal plasticity parameters ========'''
    dgamma0 = args.dgamma0
    g = args.g
    n = args.n
    c1 = args.c1
    c2 = args.c2
    b = args.b
    Q = args.Q
    H = torch.Tensor(args.H)
    theta = args.theta  
    Dglobal, slipDef, Ddemsd = D_slip(args)
    
    '''========== Start PyTorch ==========='''
    Dglobal_torch = torch.FloatTensor(Dglobal)
    slipDef_torch = torch.FloatTensor(slipDef)
    Ddemsd_torch = torch.FloatTensor(Ddemsd)
    x_train = torch.FloatTensor(data.x_train).to(device)
    y_train = torch.FloatTensor(data.y_train).to(device)
    x_validation = torch.FloatTensor(data.x_validation).to(device)
    y_validation = torch.FloatTensor(data.y_validation).to(device)
    # dgamma_validation = torch.FloatTensor(data.dgamma_validation).to(device)
        
    if args.NORM == 1:
        x_train,maxx,minx = reflectInput(x_train,args.stateID) 
        y_train,maxy,miny = reflectOutput(y_train)
        x_validation = testReflectInput(x_validation,maxx,minx,args.stateID)
        y_validation = testReflectOutput(y_validation,maxy,miny)
    
    ## set mini_batches
    torch_dataset = subDataset(x_train[:,:], y_train)
    loader = Data.DataLoader(dataset=torch_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=0,
                              )

    if args.net == 'ANN':
        ann = ANN().to(device)
    elif args.net == 'ANNsepInput':
        ann = ANNsepInput().to(device)
    elif args.net == 'ANNsepOutput':
        ann = ANNsepOutput().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ann.parameters(), lr=learning_Rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    print(ann)
    
    loss_record = []
    valLoss_record = []
    best_record = 10000
    min_Epoch = 50
    Flag = 0
    
    if RESUME:
        print("Resume from checkpoint...")
        checkpoint = torch.load(RESUME_FILE,map_location=torch.device('cpu'))     ## read previous state
        ann.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initepoch = checkpoint['epoch']+1
    else:
        initepoch = 0
     
    start_time = time.time()
    
    for epoch in range(initepoch, EPOCHS):
        for i, (data, labels) in enumerate(loader):

            if torch.cuda.is_available():  
                data = data.cuda()
                labels = labels.cuda()
            
            outputs = ann(data) 
            if args.NORM == 1:
                data = reflectBackInput(data,maxx,minx,args.stateID)
                # outputs = reflectBackOutput(outputs,maxy,miny)
            
            # ====================== loss preparation
            stress_origin = data[:,:6]
            dstrain = data[:,13:19]
            tau = torch.matmul(stress_origin,slipDef_torch.T)
            mask = torch.abs(tau) < args.r0
            
            outputs[mask] = 0
            delass = dstrain-torch.matmul(outputs,slipDef_torch)
            dstress = torch.matmul(delass,Dglobal_torch)
            stress_upd = stress_origin + dstress
            
            
            # ======================               
            if args.NORM == 1:
                stress_upd = testReflectOutput(stress_upd,maxy,miny)
            
            tau_upd = torch.matmul(stress_upd, slipDef_torch.T)
            tau_labels = torch.matmul(labels, slipDef_torch.T)
            loss_stress = criterion(tau_labels,tau_upd)
            loss = loss_stress
            optimizer.zero_grad()
            loss.backward()  # back pop
            optimizer.step()
            
            if epoch == 350:
                print(1)
            
            if (i + 1) % 1 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.8f'
                      % (epoch + 1, EPOCHS, i + 1, len(x_train) // BATCH_SIZE, loss.item()))
   
        scheduler.step()
        val_out = ann(x_validation)
        if args.NORM == 1:
            x_valid = reflectBackInput(x_validation,maxx,minx,args.stateID)        
        stress_origin = x_valid[:,:6]
        dstrain = x_valid[:,13:19]
        tau = torch.matmul(stress_origin,slipDef_torch.T)  
        mask = torch.abs(tau) < args.r0
        
        val_out[mask] = 0
        delass = dstrain-torch.matmul(val_out,slipDef_torch)
        dstress = torch.matmul(delass,Dglobal_torch)
        stress_upd = stress_origin + dstress        
        if args.NORM == 1:
            stress_upd = testReflectOutput(stress_upd,maxy,miny)
        
        val_loss = criterion(y_validation,stress_upd)
        
        if epoch > min_Epoch and best_record > val_loss.item():

            # torch.save(checkpoint, CHECKPOINT_FILE+'{}'.format(epoch))
            best_record = val_loss.item()
            checkpoint = {'epoch': epoch+1,
                'model_state_dict': ann.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':loss_record
                }
            torch.save(checkpoint, CHECKPOINT_FILE)             
            # !save the last model or the best model
        if loss.item() < 1000:
            Flag = 1
        if Flag == 1:
            loss_record.append(loss.item())
            valLoss_record.append(val_loss.item())
    
    # print('mean_train = ' + str(sum(loss_record[-100:]) / len(loss_record[-100:]))) 
    # print('mean_test = ' + str(sum(valLoss_record[-100:]) / len(valLoss_record[-100:]))) 
    
    end_time = time.time()
    print('time:' + str(end_time - start_time))
    
    plt.plot(loss_record)
    plt.plot(valLoss_record)
    plt.legend(['Training_set','Validation set'])
    plt.ylabel('loss_value')
    plt.xlabel('epochs')
    # plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return loss_record,valLoss_record


def ICMINN_train(data, args, RESUME=0):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    
    ## set hyper-parameter
    learning_Rate = args.learning_Rate
    EPOCHS = args.EPOCHS
    BATCH_SIZE = args.BATCH_SIZE
    CHECKPOINT_FILE = 'saved_model/' + args.CHECKPOINT_FILE
    RESUME_FILE = 'saved_model/' + args.RESUME_FILE
    
    '''======= Initialize crystal plasticity parameters ========'''
    dgamma0 = args.dgamma0
    g = args.g
    n = args.n
    c1 = args.c1
    c2 = args.c2
    b = args.b
    Q = args.Q
    H = torch.Tensor(args.H)
    theta = args.theta  
    Dglobal, slipDef, Ddemsd = D_slip(args)
    
    '''========== Start PyTorch ==========='''
    Dglobal_torch = torch.FloatTensor(Dglobal)
    slipDef_torch = torch.FloatTensor(slipDef)
    Ddemsd_torch = torch.FloatTensor(Ddemsd)
    if args.net == 'NN_strainRate':
        x_train = torch.FloatTensor(data.transToSlip(args)).to(device)
    elif args.net == 'ICNN_strainRate_1ele': 
        x_train = torch.FloatTensor(data.transToSlip(args)).to(device)
    elif args.net == 'NN_strainRate_decoupled': 
        x_train = torch.FloatTensor(data.transToSlip(args)).to(device)
    # x_train[:, :24] = torch.abs(x_train[:, :24])
    y_train = torch.FloatTensor(data.label).to(device)
        
    if args.NORM == 1:
        x_train,maxx,minx = reflectInput(x_train,args.stateID) 
        y_train,maxy,miny = reflectOutput(y_train)
    
    ## set mini_batches
    torch_dataset = subDataset(x_train[:,:], y_train)
    loader = Data.DataLoader(dataset=torch_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=0,
                              )

    if args.net == 'ANN':
        ann = ANN().to(device)
    elif args.net == 'NN_strainRate': 
        ann = NN_strainRate().to(device)
    elif args.net == 'ICNN_strainRate_1ele': 
        ann = ICNN_strainRate_1ele().to(device)        
    elif args.net == 'NN_strainRate_decoupled': 
        ann = NN_strainRate_decoupled().to(device)        
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(ann.parameters(), lr=learning_Rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    print(ann)
    
    loss_record = []
    valLoss_record = []
    best_record = 10000  
    min_Epoch = 50
    Flag = 0
    
    if RESUME:
        print("Resume from checkpoint...")
        checkpoint = torch.load(RESUME_FILE,map_location=torch.device('cpu'))     ## read previous state
        ann.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        initepoch = checkpoint['epoch']+1
    else:
        initepoch = 0
     
    start_time = time.time()
    
    for epoch in range(initepoch, EPOCHS):
        for i, (data, labels) in enumerate(loader):

            if torch.cuda.is_available():  
                data = data.cuda()
                labels = labels.cuda()
            
            outputs = ann(data) 
            if args.NORM == 1:
                data = reflectBackInput(data,maxx,minx,args.stateID)
                # outputs = reflectBackOutput(outputs,maxy,miny)
            
            # ====================== loss preparation
            stress_origin = data[:,args.stateID[2]]
            tau = torch.matmul(stress_origin, slipDef_torch.T)
            # tau_signs = torch.sign(tau)
            dstrain = data[:,args.stateID[3]]
            mask = torch.abs(tau) < args.r0
               
            outputs[mask] = 0
            # outputs = torch.abs(outputs) * tau_signs
            
            # outputs = outputs * tau_signs
            delass = dstrain-torch.matmul(outputs,slipDef_torch)
            dstress = torch.matmul(delass,Dglobal_torch)
            
            stress_upd = stress_origin + dstress
            
            
            # ======================               
            if args.NORM == 1:
                stress_upd = testReflectOutput(stress_upd,maxy,miny)
            
            # tau_upd = torch.matmul(stress_upd, slipDef_torch.T)
            # tau_labels = torch.matmul(labels, slipDef_torch.T)
            # loss_stress = criterion(tau_labels,tau_upd)
            loss_stress = criterion(labels,stress_upd)
            loss = loss_stress
            optimizer.zero_grad()  
            loss.backward()  # back pop
            optimizer.step()
            ann.apply_constraints() # enforce convex constraints
            
            if epoch == 350:
                print(1)
            
            if (i + 1) % 1 == 0:
                print('Epoch [%d/%d], Iter [%d/%d] Loss: %.8f'
                      % (epoch + 1, EPOCHS, i + 1, len(x_train) // BATCH_SIZE, loss.item()))
   
        scheduler.step()
        
        if epoch > min_Epoch and best_record > loss.item():

            # torch.save(checkpoint, CHECKPOINT_FILE+'{}'.format(epoch))
            best_record = loss.item()
            checkpoint = {'epoch': epoch+1,
                'model_state_dict': ann.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':loss_record
                }
            torch.save(checkpoint, CHECKPOINT_FILE)             
            # !save the last model or the best model
        if loss.item() < 1000:
            Flag = 1
        if Flag == 1:
            loss_record.append(loss.item())
    
    # print('mean_train = ' + str(sum(loss_record[-100:]) / len(loss_record[-100:]))) 
    # print('mean_test = ' + str(sum(valLoss_record[-100:]) / len(valLoss_record[-100:]))) 
    
    end_time = time.time()
    print('time:' + str(end_time - start_time))
    
    plt.plot(loss_record)
    plt.legend(['Training_set'])
    plt.ylabel('loss_value')
    plt.xlabel('epochs')
    # plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return loss_record




if __name__ == "__main__":
    pass   