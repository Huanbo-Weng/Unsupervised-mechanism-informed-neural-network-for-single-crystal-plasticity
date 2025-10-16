import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import h5py
from torch.optim import lr_scheduler
from torch.utils.data.dataset import Dataset
from functions import *
from MINN_RVE import *
from ICMINN import *
import time

def preTest(data, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_FILE = 'saved_model/' + args.RESUME_FILE
    x_train = torch.FloatTensor(data.x_train)
    y_train = torch.FloatTensor(data.y_train)
    x_test = torch.FloatTensor(data.x_test).to(device)
    y_test = torch.FloatTensor(data.y_test).to(device)
    if args.net == 'ANN':
        ann = ANN().to(device)
    elif args.net == 'ANNsepInput':
        ann = ANNsepInput().to(device)
    elif args.net == 'ANNsepOutput':
        ann = ANNsepOutput().to(device)

    if args.NORM == 1:
        x_train,maxx,minx = reflectInput(x_train,args.stateID) 
        y_train,maxy,miny = reflectOutput(y_train)
        x_test = testReflectInput(x_test,maxx,minx,args.stateID)
        

    print("Resume from checkpoint...")
    checkpoint = torch.load(CHECKPOINT_FILE,map_location='cpu')   ## read previous state
    ann.load_state_dict(checkpoint['model_state_dict'])

    outputs = ann(x_test)
    if args.NORM == 1:
        outputs = reflectBackOutput(outputs,maxy,miny)

    outputs = outputs.cpu().detach().numpy()
    y_test = y_test.cpu().detach().numpy()

    '''========== Visualization and evaluation ==========='''
    plt.figure(1)
    min_value = min(outputs.min(), y_test.min())
    max_value = max(outputs.max(), y_test.max())    
    plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--')
    plt.scatter(outputs,y_test)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
    print(f"MSE：{mean_squared_error(outputs, y_test)}")
    print(f"MAE：{mean_absolute_error(outputs, y_test)}")
    print(f"MAPE：{mean_absolute_percentage_error(outputs, y_test)}")
    print(f"R^2：{r2_score(outputs, y_test)}")    

    return outputs


def evalTrain(data, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_FILE = 'saved_model/' + args.RESUME_FILE
    x_train = torch.FloatTensor(data.x_train)
    y_train = torch.FloatTensor(data.y_train)
    # dgamma_train = data.dgamma_train
    if args.net == 'ANN':
        ann = ANN().to(device)
    elif args.net == 'ANNsepInput':
        ann = ANNsepInput().to(device)
    elif args.net == 'ANNsepOutput':
        ann = ANNsepOutput().to(device)
    '''========== Read parameters ==========='''
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
    Dglobal_torch = torch.FloatTensor(Dglobal)
    slipDef_torch = torch.FloatTensor(slipDef)
    Ddemsd_torch = torch.FloatTensor(Ddemsd)

    if args.NORM == 1:
        x_train,maxx,minx = reflectInput(x_train,args.stateID) 

    print("Resume from checkpoint...")
    checkpoint = torch.load(CHECKPOINT_FILE,map_location='cpu')   ## read previous state
    ann.load_state_dict(checkpoint['model_state_dict'])

    outputs = ann(x_train)

    if args.NORM == 1:
        x_train = reflectBackInput(x_train,maxx,minx,args.stateID)

    '''========== Calculate physics-informed dgamma ==========='''
    stress_origin = x_train[:,:6]
    dstrain = x_train[:,13:19]
    tau = torch.matmul(stress_origin,slipDef_torch.T)
    mask = torch.abs(tau) < args.r0
    
    outputs[mask] = 0
    dplass = torch.matmul(outputs,slipDef_torch)
    # dplass = dplass.cpu().detach().numpy()
    # dplass_true = np.dot(dgamma_test,slipDef)
    delass = dstrain-dplass
    dstress = torch.matmul(delass,Dglobal_torch)
    stress_upd = stress_origin + dstress

    outputs = outputs.cpu().detach().numpy()
    stress_upd = stress_upd.cpu().detach().numpy()
    y_train = y_train.cpu().detach().numpy()

    '''========== Visualization and evaluation ==========='''
    plt.figure(1)
    min_value = min(stress_upd.min(), y_train.min())
    max_value = max(stress_upd.max(), y_train.max())    
    plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--')
    plt.scatter(stress_upd,y_train)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
    print(f"MSE：{mean_squared_error(stress_upd, y_train)}")
    print(f"MAE：{mean_absolute_error(stress_upd, y_train)}")
    print(f"MAPE：{mean_absolute_percentage_error(stress_upd, y_train)}")
    print(f"R^2：{r2_score(stress_upd, y_train)}")
    
    outputs = outputs[:,[0,1,3,5,6,8,9,10]]
    # dgamma_train = dgamma_train[:,[0,1,3,5,6,8,9,10]]
    # plt.figure(2)
    # min_value = min(outputs.min(), dgamma_train.min())
    # max_value = max(outputs.max(), dgamma_train.max())    
    # plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--')
    # plt.scatter(outputs,dgamma_train)      

    return outputs, stress_upd


def test_StrainRate(data, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_FILE = 'saved_model/' + args.RESUME_FILE
    if args.net == 'NN_strainRate':
        x_train = torch.FloatTensor(data.transToSlip(args)).to(device)
    elif args.net == 'NN_strainRate_1ele': 
        x_train = torch.FloatTensor(data.transToSlip(args)).to(device)
    elif args.net == 'ICNN_strainRate_1ele': 
        x_train = torch.FloatTensor(data.transToSlip(args)).to(device)
    elif args.net == 'NN_strainRate_decoupled': 
        x_train = torch.FloatTensor(data.transToSlip(args)).to(device)
    # x_train[:, :24] = torch.abs(x_train[:, :24])
    y_train = torch.FloatTensor(data.label).to(device)
    if args.net == 'ANN':
        ann = ANN().to(device)
    elif args.net == 'NN_strainRate': 
        ann = NN_strainRate().to(device)
    elif args.net == 'NN_strainRate_1ele': 
        ann = NN_strainRate_1ele().to(device)  
    elif args.net == 'ICNN_strainRate_1ele':
        ann = ICNN_strainRate_1ele().to(device)
    elif args.net == 'NN_strainRate_decoupled': 
        ann = NN_strainRate_decoupled().to(device)  
    '''========== Read parameters ==========='''
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
    Dglobal_torch = torch.FloatTensor(Dglobal)
    slipDef_torch = torch.FloatTensor(slipDef)
    Ddemsd_torch = torch.FloatTensor(Ddemsd)

    if args.NORM == 1:
        x_train,maxx,minx = reflectInput(x_train,args.stateID) 

    print("Resume from checkpoint...")
    checkpoint = torch.load(CHECKPOINT_FILE,map_location='cpu')   ## read previous state
    ann.load_state_dict(checkpoint['model_state_dict'])

    outputs = ann(x_train)

    if args.NORM == 1:
        x_train = reflectBackInput(x_train,maxx,minx,args.stateID)

    '''========== Calculate physics-informed dgamma ==========='''
    stress_origin = x_train[:,args.stateID[2]]
    tau = torch.matmul(stress_origin, slipDef_torch.T)
    tau_signs = torch.sign(tau)        
    dstrain = x_train[:,args.stateID[3]]
    mask = torch.abs(tau) < args.r0
    outputs[mask] = 0
    # outputs = torch.abs(outputs) * tau_signs
    delass = dstrain-torch.matmul(outputs,slipDef_torch)
    dstress = torch.matmul(delass,Dglobal_torch)
    
    stress_upd = stress_origin + dstress

    outputs = outputs.cpu().detach().numpy()
    stress_upd = stress_upd.cpu().detach().numpy()
    y_train = y_train.cpu().detach().numpy()
    
    # Since values near 0 make it difficult for R-square to reflect the overall goodness of fit
    # they can be excluded during the evaluation.
    y_train[np.abs(y_train)<1e-2] = 0
    stress_upd[np.abs(y_train)<1] = 0     

    '''========== Visualization and evaluation ==========='''
    plt.figure(1, figsize=(8, 6))  # Set the figure size
    
    # Find the minimum and maximum values
    min_value = min(stress_upd.min(), y_train.min())
    max_value = max(stress_upd.max(), y_train.max())
    
    # Plot the diagonal line
    plt.plot([min_value, max_value], [min_value, max_value], color='red', linestyle='--', label='Ideal Prediction')
    
    # Plot the scatter plot
    plt.scatter(stress_upd, y_train, label='Data Points', alpha=0.6)  # Add transparency
    
    # Add title and axis labels
    plt.xlabel('MINN Predicted Stress (MPa)', fontsize=14)
    plt.ylabel('FEM Simulated Stress (MPa)', fontsize=14)
    
    # Add legend
    plt.legend(fontsize=12)
    
    # Add grid lines
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Set axis ranges
    plt.xlim(min_value, max_value)
    plt.ylim(min_value, max_value)
    
    # Set tick label sizes
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    # Display the figure
    plt.show()
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
    print(f"MSE: {mean_squared_error(stress_upd, y_train)}")
    print(f"MAE: {mean_absolute_error(stress_upd, y_train)}")
    print(f"MAPE: {mean_absolute_percentage_error(stress_upd, y_train)}")
    print(f"R^2: {r2_score(stress_upd, y_train)}")

    return outputs, stress_upd


def test_gradient(data, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT_FILE = 'saved_model/' + args.RESUME_FILE
    
    # Load input data and set requires_grad=True to enable gradient calculation
    if args.net == 'NN_strainRate':
        x_train = torch.FloatTensor(data.transToSlip(args)).to(device)
    elif args.net == 'NN_strainRate_1ele': 
        x_train = torch.FloatTensor(data.transToSlip(args)).to(device)
    elif args.net == 'ICNN_strainRate_1ele': 
        x_train = torch.FloatTensor(data.transToSlip(args)).to(device)
    elif args.net == 'NN_strainRate_decoupled': 
        x_train = torch.FloatTensor(data.transToSlip(args)).to(device)
    
    # Enable gradient computation
    x_train.requires_grad_(True)
    
    y_train = torch.FloatTensor(data.label).to(device)
    
    # Initialize the model
    if args.net == 'ANN':
        ann = ANN().to(device)
    elif args.net == 'NN_strainRate': 
        ann = NN_strainRate().to(device)
    elif args.net == 'NN_strainRate_1ele': 
        ann = NN_strainRate_1ele().to(device)  
    elif args.net == 'ICNN_strainRate_1ele':
        ann = ICNN_strainRate_1ele().to(device)
    elif args.net == 'NN_strainRate_decoupled': 
        ann = NN_strainRate_decoupled().to(device)  
    
    # Normalization processing
    if args.NORM == 1:
        x_train, maxx, minx = reflectInput(x_train, args.stateID) 

    # Load model weights
    print("Resume from checkpoint...")
    checkpoint = torch.load(CHECKPOINT_FILE, map_location='cpu')
    ann.load_state_dict(checkpoint['model_state_dict'])

    # Model inference
    outputs = ann(x_train)

    # Set grad_outputs to ones with the same shape as outputs, indicating derivative calculation with respect to outputs themselves
    grad = torch.autograd.grad(
        outputs=outputs,
        inputs=x_train,
        grad_outputs=torch.ones_like(outputs),
        create_graph=False,  # No need to create computation graph in evaluation mode
        retain_graph=False,
        only_inputs=True
    )[0]  # Return the first element of the gradient tuple (i.e., gradient of x_train)
    grad = grad.cpu().detach().numpy()
    
    # Return outputs, predicted stress, and gradients for each sample
    return grad


    