# -*- coding: utf-8 -*-

# Import modules ##############################################################
import numpy as np
import Black_Scholes as bs
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import qmc
from scipy.spatial import KDTree
import statistics
import time



# Global Problem Constants #####################################################


"""
Create a range of non-dimensional asset prices and times [yrs] by normalising 
with respect to K (strike price) and T (time to expiry [yrs]) - both are constant

r and sigma need to be added as inputs to the NN for a true universal option pricing model.
"""   
MAX_S_factor = 3
K = 1              #non-dimensional strike price
T = 1              #non-dimensional strike price

# Configuration constants ######################################################

N_DATA_POINTS = 2000
N_COLLOCATION_POINTS = 2000
N_BC_POINTS_PER_BOUNDARY = 100
N_BC_POINTS_TOTAL = N_BC_POINTS_PER_BOUNDARY * 3   # S=0, S=Smax, terminal condition
HIDDEN_DIM = 32
TOTAL_EPOCHS = 8000
OPTIMIZER_SWITCH_EPOCH = 4000   # epoch at which training switches from Adam to LBFGS


# Generate Training Data + Collocation Points ##################################

def generate_data(seed=None, plot=True):
    """
    Generates the training (data) points and the Latin-hypercube collocation
    points used by the rest of the script. Optionally seeded so that repeat
    calls (e.g. across seed_data_analytics samples) draw an independent
    Latin hypercube each time rather than reusing the same fixed dataset.
    """

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Data points
    sampler = qmc.LatinHypercube(d=4)
    sample = sampler.random(n=N_DATA_POINTS)            
    sample[:,0] = sample[:,0]*MAX_S_factor
    S_flat = sample[:, 0]
    t_flat = sample[:, 1]
    r_flat = sample[:, 2] * 0.1                 #risk free rate in the range [0, 0.1]
    sigma_flat = sample[:, 3] * 0.6             #volatilty in the range [0, 0.6]

    tau_flat = T - t_flat

    V_flat = np.array([
        bs.V_i_call(s, K) if tau == 0
        else bs.ComputeClosedFormCall(s, K, r, sigma, tau)
        for s, tau, r, sigma in zip(S_flat, tau_flat, r_flat, sigma_flat)
    ])

    inputs = np.column_stack((S_flat, t_flat, r_flat, sigma_flat))   # shape = (2000, 2)
    targets = V_flat.reshape(-1, 1) # shape = (2000, 1) - 2000 rows one column

    X_train = torch.tensor(inputs, dtype=torch.float32)   # shape = (2000, 2)
    Y_train = torch.tensor(targets, dtype=torch.float32)  # shape = (2000, 1)

    # Collocation points
    sampler = qmc.LatinHypercube(d=4)
    sample = sampler.random(n=N_COLLOCATION_POINTS)

    sample[:,0] = sample[:,0]*MAX_S_factor
    sample[:,2] = sample[:,2] * 0.1
    sample[:,3] = sample[:,3] * 0.6

    S_CollocationPoints = torch.tensor(sample[:,0].reshape(-1,1), dtype=torch.float32, requires_grad=True)
    t_CollocationPoints = torch.tensor(sample[:,1].reshape(-1,1), dtype=torch.float32, requires_grad=True)
    r_CollocationPoints = torch.tensor(sample[:,2].reshape(-1,1), dtype=torch.float32, requires_grad=False)
    sig_CollocationPoints = torch.tensor(sample[:,3].reshape(-1,1), dtype=torch.float32, requires_grad=False)

    if plot:
        plt.scatter(sample[:,0].reshape(-1,1), sample[:,1].reshape(-1,1), marker ='x', color='tab:red')
        plt.title("Collocation Points - Latin Hypercube")
        plt.xlabel("$\\hat{S}$")
        plt.ylabel("$\\hat{t}$")
        plt.grid()
        plt.show()

    return X_train, Y_train, S_CollocationPoints, t_CollocationPoints, r_CollocationPoints, sig_CollocationPoints


# Define PINN Network Class in PyTorch ########################################

class PINN_MLP(nn.Module):
    def __init__(self): 
        super().__init__()
        self.net = nn.Sequential(        
            nn.Linear(4, HIDDEN_DIM),             #2 inputs connected to 32 neurons
            nn.Tanh(),                   

            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),         
            nn.Tanh(),
            
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM),          
            nn.Tanh(),

            nn.Linear(HIDDEN_DIM, 1),            #output layer maps 32 neurons as inputs to a singular scalar value = predicted next day log return
        )

    def forward(self, x):
        return self.net(x)


# Define PDE Residual Loss Function ###########################################


def PDE_Residual_Loss(model, S_CollocationPoints, t_CollocationPoints, r_CollocationPoints, sig_CollocationPoints):
    
    CollocationPoints = torch.cat([S_CollocationPoints, t_CollocationPoints, r_CollocationPoints, sig_CollocationPoints], dim=1)
    V_colPoint = model(CollocationPoints)
    
    dV_dS = torch.autograd.grad(V_colPoint, S_CollocationPoints, grad_outputs=torch.ones_like(V_colPoint), create_graph=True)[0]
    d2V_dS2 = torch.autograd.grad(dV_dS, S_CollocationPoints, grad_outputs=torch.ones_like(dV_dS), create_graph=True)[0]
    dV_dt = torch.autograd.grad(V_colPoint, t_CollocationPoints, grad_outputs=torch.ones_like(V_colPoint), create_graph=True)[0]
    
    error = dV_dt + 0.5*sig_CollocationPoints**2 *S_CollocationPoints**2 *d2V_dS2 + r_CollocationPoints*S_CollocationPoints*dV_dS - r_CollocationPoints*V_colPoint
    MSE = torch.mean(error**2)

    return MSE, (error**2).detach()

    
# Define BC and TC Loss Function ##############################################
def BC_Loss_Call(model, T):
    
    tdivT_BC = np.linspace(0, 1, num=N_BC_POINTS_PER_BOUNDARY)
    tdivT_TC = np.full(N_BC_POINTS_PER_BOUNDARY, 1)
    
    SdivK_BC_0 = np.zeros(len(tdivT_BC))
    SdivK_BC_max = np.full(len(tdivT_BC), MAX_S_factor)
    SdivK_BC = np.linspace(0, MAX_S_factor, num=N_BC_POINTS_PER_BOUNDARY)
    
    r = np.linspace(0, 0.1, num=N_BC_POINTS_PER_BOUNDARY)
    sigma = np.linspace(0, 0.6, num=N_BC_POINTS_PER_BOUNDARY)
    
    S0_inputs = np.column_stack((SdivK_BC_0, tdivT_BC, r, sigma))
    Smax_inputs = np.column_stack((SdivK_BC_max, tdivT_BC, r, sigma))
    Srand_inputs = np.column_stack((SdivK_BC, tdivT_TC, r, sigma))
    
    S0_inputs = torch.tensor(S0_inputs, dtype=torch.float32)
    Smax_inputs = torch.tensor(Smax_inputs, dtype=torch.float32)
    Srand_inputs = torch.tensor(Srand_inputs, dtype=torch.float32)
    
    V_BC0 = model(S0_inputs)
    errors_0 = V_BC0
    
    V_BCmax = model(Smax_inputs)
    V_true_max = torch.tensor(
        (SdivK_BC_max - K*np.exp(-r*(T-tdivT_BC))).reshape(-1,1),
        dtype=torch.float32)
    errors_max = V_BCmax - V_true_max
    
    V_TC = model(Srand_inputs)
    V_true_expiry = torch.clamp(
        torch.tensor(SdivK_BC - 1, dtype=torch.float32).reshape(-1,1),
        min=0)
    errors_expiry = V_TC - V_true_expiry
    
    errors_total = torch.cat((errors_0, errors_max, errors_expiry))
    
    return errors_total


# Define Training Loop ########################################################

def Training_Loop(pecann_params, X_train, Y_train, S_CollocationPoints, t_CollocationPoints, r_CollocationPoints, sig_CollocationPoints, adapt, adapt_params={}, lossOutputPeriod=5):
            
    def closure():
        optimizer.zero_grad()
        Data_loss = (model(X_train) - Y_train)
        PDE_loss, PDE_residuals = PDE_Residual_Loss(model, S_CollocationPoints, t_CollocationPoints, r_CollocationPoints, sig_CollocationPoints)
        BC_loss = BC_Loss_Call(model, T)
        ALM_loss = PDE_loss + torch.mean(lambda_BC * BC_loss) + 0.5*mu_BC*(torch.mean(BC_loss**2)) + torch.mean(lambda_data * Data_loss) + 0.5*mu_data*(torch.mean(Data_loss**2))
        ALM_loss.backward()
        return ALM_loss
    
    Data_losses, PDE_losses, BC_losses = np.array([]), np.array([]), np.array([])
    model = PINN_MLP()
    k = pecann_params["k"]
    beta = pecann_params["beta"]
    gamma = pecann_params["gamma"]
    mu_max = pecann_params["max mu"]
    start_epoch = pecann_params["starting epoch"]
    
    lambda_BC = torch.ones((N_BC_POINTS_TOTAL, 1))
    lambda_data = torch.ones((N_DATA_POINTS, 1))
    mu_BC = torch.tensor(1.0)     
    mu_data = torch.tensor(1.0) 
    Data_loss_prev = torch.zeros((N_DATA_POINTS, 1))
    BC_loss_prev = torch.zeros((N_BC_POINTS_TOTAL, 1))
    
    mu_BC_list, mu_data_list, bc_ratio, data_ratio = [], [], [1.0], [1.0]
    mu_BC_list.append(1.0)
    mu_data_list.append(1.0)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    
    for epoch in range(TOTAL_EPOCHS + 1):
        
        if epoch == OPTIMIZER_SWITCH_EPOCH:
            optimizer = optim.LBFGS(model.parameters(), lr=1.0, max_iter=10, history_size=20, line_search_fn='strong_wolfe')
        
        
        model.train()
        optimizer.zero_grad()
        
        Data_loss = (model(X_train) - Y_train)
        PDE_loss, PDE_residuals = PDE_Residual_Loss(model, S_CollocationPoints, t_CollocationPoints, r_CollocationPoints, sig_CollocationPoints)
        BC_loss = BC_Loss_Call(model, T)
        
        if epoch % k == 0 and epoch != 0 and epoch >= start_epoch:
            if torch.norm(Data_loss) > (gamma * torch.norm(Data_loss_prev)):
                if mu_data < mu_max:
                    mu_data = beta * mu_data
                
                
            if torch.norm(BC_loss) > (gamma * torch.norm(BC_loss_prev)):
                if mu_BC < mu_max:
                    mu_BC = beta * mu_BC
                
                
            data_ratio.append((torch.norm(Data_loss)/torch.norm(Data_loss_prev)).detach())
            bc_ratio.append((torch.norm(BC_loss)/torch.norm(BC_loss_prev)).detach())
            mu_BC_list.append(mu_BC)
            mu_data_list.append(mu_data)
                
            lambda_BC_new = lambda_BC + mu_BC * BC_loss.detach()
            lambda_data_new = lambda_data + mu_data * Data_loss.detach()
            lambda_BC = lambda_BC_new
            lambda_data = lambda_data_new
            
            Data_loss_prev = Data_loss.detach()
            BC_loss_prev = BC_loss.detach()
        
        ALM_loss = PDE_loss + torch.mean(lambda_BC * BC_loss) + 0.5*mu_BC*(torch.mean(BC_loss**2)) + torch.mean(lambda_data * Data_loss) + 0.5*mu_data*(torch.mean(Data_loss**2))
        
        if adapt == True:
            S_CollocationPoints, t_CollocationPoints, r_CollocationPoints, sig_CollocationPoints = collocation_adapt(adapt_params, PDE_residuals, epoch, S_CollocationPoints, t_CollocationPoints, r_CollocationPoints, sig_CollocationPoints)
        
        if epoch % lossOutputPeriod == 0:
            print(f"epoch: {epoch}")
            Data_losses = np.append(Data_losses, torch.mean(Data_loss**2).detach().item())
            PDE_losses = np.append(PDE_losses, PDE_loss.detach().item())
            BC_losses = np.append(BC_losses, torch.mean(BC_loss**2).detach().item())
            
        if epoch >= OPTIMIZER_SWITCH_EPOCH:
            optimizer.step(closure)
        else:
            ALM_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
    mu = [mu_data_list, mu_BC_list]
    ratios = [data_ratio, bc_ratio]
    return Data_losses, PDE_losses, BC_losses, model, mu, ratios
 
    
    
# Train the Model #############################################################
def train(pecann_params, adapt, adapt_params, lossOutputPeriod, X_train, Y_train, S_CollocationPoints, t_CollocationPoints, r_CollocationPoints, sig_CollocationPoints):
    
    Data_losses, PDE_losses, BC_losses, model, mu, ratios = Training_Loop(
        pecann_params, X_train, Y_train, S_CollocationPoints, t_CollocationPoints, r_CollocationPoints,
        sig_CollocationPoints, adapt, adapt_params, lossOutputPeriod)
    
    epochs=np.arange(0, TOTAL_EPOCHS, step=lossOutputPeriod)
    epochs = np.append(epochs, TOTAL_EPOCHS)
    
    plt.plot(epochs, Data_losses, 'r-', label='$L_{Data}(\\theta)$')
    plt.plot(epochs, PDE_losses, 'b-', label='$L_{PDE}(\\theta)$')
    plt.plot(epochs, BC_losses, 'g-', label='$L_{BC}(\\theta)$')
    plt.semilogy()
    plt.legend()
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("$L(\\theta)$")
    plt.show()
    
    epochs_mu = np.arange(0, TOTAL_EPOCHS, step=50)
    epochs_mu = np.append(epochs_mu, TOTAL_EPOCHS)
    plt.plot(epochs_mu, mu[0], 'r-', label='$\\mu_{Data}$')
    plt.plot(epochs_mu, mu[1], 'b-', label='$\\mu_{BC}$')
    plt.legend()
    plt.grid()
    plt.semilogy()
    plt.xlabel("epoch")
    plt.ylabel("$\\mu$")
    plt.show()
 
    plt.plot(epochs_mu, ratios[0], 'r-', label='Data')
    plt.plot(epochs_mu, ratios[1], 'b-', label='BC')
    plt.legend()
    plt.grid()
    plt.xlabel("epoch")
    plt.ylabel("$\\frac{R(\\theta_{k})}{R(\\theta_{k-1})}$")
    plt.show()
    
    return model
        
# Validate the Model ##########################################################

def validate(model):
    
    model.eval()
    
    r_eval_low = 0.01
    sigma_eval_low = 0.05
    
    r_eval_mid = 0.05
    sigma_eval_mid = 0.3
    
    r_eval_high = 0.09
    sigma_eval_high = 0.55
    
    SdivK_eval = np.linspace(0, MAX_S_factor, 200)
    tdivT_eval = np.full(len(SdivK_eval), 0)
    r_eval_array_low = np.full(len(SdivK_eval), r_eval_low)
    sigma_eval_array_low = np.full(len(SdivK_eval), sigma_eval_low)
    r_eval_array_mid = np.full(len(SdivK_eval), r_eval_mid)
    sigma_eval_array_mid = np.full(len(SdivK_eval), sigma_eval_mid)
    r_eval_array_high = np.full(len(SdivK_eval), r_eval_high)
    sigma_eval_array_high = np.full(len(SdivK_eval), sigma_eval_high)
    
    inputs_eval_low = np.column_stack((SdivK_eval, tdivT_eval, r_eval_array_low, sigma_eval_array_low))
    inputs_eval_low = torch.tensor(inputs_eval_low, dtype=torch.float32)
    inputs_eval_mid = np.column_stack((SdivK_eval, tdivT_eval, r_eval_array_mid, sigma_eval_array_mid))
    inputs_eval_mid = torch.tensor(inputs_eval_mid, dtype=torch.float32)
    inputs_eval_high = np.column_stack((SdivK_eval, tdivT_eval, r_eval_array_high, sigma_eval_array_high))
    inputs_eval_high = torch.tensor(inputs_eval_high, dtype=torch.float32)
    
    V_eval_low = model(inputs_eval_low)
    V_eval_low = V_eval_low.detach().numpy().flatten()  #convert from PyTorch tensor to numpy array of shape (200,)
    V_eval_mid = model(inputs_eval_mid)
    V_eval_mid = V_eval_mid.detach().numpy().flatten()  #convert from PyTorch tensor to numpy array of shape (200,)
    V_eval_high = model(inputs_eval_high)
    V_eval_high = V_eval_high.detach().numpy().flatten()  #convert from PyTorch tensor to numpy array of shape (200,)
    
    V_true_low, V_true_mid, V_true_high = np.array([]), np.array([]), np.array([])
    tau = 1    #T -t/T = 1-0 = 1  (since t=0 at current day)
    
    for i in SdivK_eval:
       V_true_low = np.append(V_true_low, bs.ComputeClosedFormCall(i, K, r_eval_low, sigma_eval_low, tau))
       V_true_mid = np.append(V_true_mid, bs.ComputeClosedFormCall(i, K, r_eval_mid, sigma_eval_mid, tau))
       V_true_high = np.append(V_true_high, bs.ComputeClosedFormCall(i, K, r_eval_high, sigma_eval_high, tau))
     
    plt.plot(SdivK_eval, V_true_mid, 'r-', label='Closed-Form')
    plt.plot(SdivK_eval, V_eval_mid, 'b--', label='PECANN + adapt')
    plt.xlabel("$\\hat{S}$")
    plt.ylabel("$\\hat{V}$")
    plt.grid()
    plt.title("Current Day Payoff Curve ($\\sigma_{mid}, r_{mid}$)")
    plt.legend()
    plt.show()
    
    error_low = np.abs(V_eval_low - V_true_low)  #/(np.abs(V_true) + 1e-5)
    error_mid = np.abs(V_eval_mid - V_true_mid)  #/(np.abs(V_true) + 1e-5)
    error_high = np.abs(V_eval_high - V_true_high)  #/(np.abs(V_true) + 1e-5)
    
    plt.plot(SdivK_eval, error_low, 'r-', label='low')
    plt.plot(SdivK_eval, error_mid, 'b-', label = 'mid')
    plt.plot(SdivK_eval, error_high, 'g-', label = 'high')
    plt.xlabel("$\\hat{S}$")
    plt.ylabel("Absolute Error to Closed-Form Solution")
    plt.semilogy()
    plt.grid()
    plt.legend()
    plt.show()
    
    t_hat = np.linspace(0, 0.99, 15)
    RMSE = []
    
    for i in range(len(t_hat)):
        tdivT_eval = np.full(len(SdivK_eval), t_hat[i])
        inputs_eval_mid = np.column_stack((SdivK_eval, tdivT_eval, r_eval_array_mid, sigma_eval_array_mid))
        inputs_eval_mid = torch.tensor(inputs_eval_mid, dtype=torch.float32)
        V_eval_mid = model(inputs_eval_mid)
        V_eval_mid = V_eval_mid.detach().numpy().flatten()  #convert from PyTorch tensor to numpy array of shape (200,)
        V_true_mid = np.array([])
        for s_val in SdivK_eval:
            V_true_mid = np.append(V_true_mid, bs.ComputeClosedFormCall(s_val, K, r_eval_mid, sigma_eval_mid, 1-t_hat[i]))
        
        RMSE.append(np.sqrt(np.mean(np.abs(V_eval_mid - V_true_mid)**2)))
    
    plt.plot(t_hat, RMSE, 'k-')
    plt.xlabel("$\\hat{t}$")
    plt.ylabel("RMSE")
    plt.grid()
    plt.show()
    
    
    print(f"RMSE to Closed-Form Across S/K Grid (low) = {(np.sqrt(np.mean(error_low**2))):.3e}")
    print(f"RMSE to Closed-Form Across S/K Grid (mid) = {(np.sqrt(np.mean(error_mid**2))):.3e}")
    print(f"RMSE to Closed-Form Across S/K Grid (high) = {(np.sqrt(np.mean(error_high**2))):.3e}")
    
    return (np.sqrt(np.mean(error_low**2))), (np.sqrt(np.mean(error_mid**2))), (np.sqrt(np.mean(error_high**2)))
    
# Define Adaptive Collocation Point Refinement Function #######################

def collocation_adapt(adapt_params, PDE_residuals, epoch, S_CollPts, t_CollPts, r_CollPts, sig_CollPts):
    
    start_epoch = adapt_params["starting epoch"]
    max_coll = adapt_params["maximum collocation points"]
    add_points = adapt_params["per adaptation added points"]
    adapt_period = adapt_params["adaptation period"]
    resid_thresh = adapt_params["residual threshold"]
    
    
    if epoch >= start_epoch and (epoch - start_epoch) % adapt_period == 0 and len(S_CollPts) < max_coll:
        
        S_CollPts = S_CollPts.detach().squeeze(1).numpy()
        t_CollPts = t_CollPts.detach().squeeze(1).numpy()
        r_CollPts = r_CollPts.detach().squeeze(1).numpy()
        sig_CollPts = sig_CollPts.detach().squeeze(1).numpy()
        CollPts = np.column_stack((S_CollPts, t_CollPts, r_CollPts, sig_CollPts))
        
        Smin, Smax = min(S_CollPts), max(S_CollPts) #defne domain boundaries
        tmin, tmax = 0, 1
        rmin, rmax = 0, 0.1
        sigmin, sigmax = 0, 0.6
        
        # Plot Residual Contour over S,t grid #################################
        plt.figure(figsize=(8,6))
        cont = plt.tricontourf(S_CollPts, t_CollPts, PDE_residuals.flatten(), levels=50, cmap='viridis')
        plt.colorbar(cont, label='Residual')
        plt.xlabel("$\\hat{S}$")
        plt.ylabel("$\\hat{t}$")
        plt.title(f'Collocation Point Residual,  epoch = {epoch}')
        plt.show()
        #######################################################################
        
        adaptPts_idxs = np.where(PDE_residuals.flatten() > resid_thresh)[0]

        tree = KDTree(CollPts)
        t_newPts, S_newPts, r_newPts, sig_newPts = np.array([]), np.array([]), np.array([]), np.array([])
        d = 4
        
        for pnt in adaptPts_idxs:
            query = np.array([S_CollPts[pnt], t_CollPts[pnt], r_CollPts[pnt], sig_CollPts[pnt]], dtype=np.float32)
            # k=2 the nearest match to a point already in the tree is always
            # itself at distance 0, so need 2nd nearest neighbour
            # to get non-zero refinement radius.
            dist, idx = tree.query(query, k=2)
            radius = dist[1]
        
            # Gaussian-normalise hypersphere sampling
            g = np.random.normal(size=(add_points, d))
            unit_dir = g / np.linalg.norm(g, axis=1, keepdims=True)   # uniform direction on S^3
        
            u = qmc.LatinHypercube(d=1).random(n=add_points).flatten()
            r = radius * u**(1/d)                                     # uniform-by-volume radius
        
            offsets = r[:, None] * unit_dir                            # shape (add_points, 4)
        
            Spts   = S_CollPts[pnt]   + offsets[:, 0]                  # Gaussian normalise
            tpts   = t_CollPts[pnt]   + offsets[:, 1]                  # Isotropic gaussian density
            rpts   = r_CollPts[pnt]   + offsets[:, 2]
            sigpts = sig_CollPts[pnt] + offsets[:, 3]
        
            S_newPts = np.append(S_newPts, Spts)
            t_newPts = np.append(t_newPts, tpts)
            r_newPts = np.append(r_newPts, rpts)
            sig_newPts = np.append(sig_newPts, sigpts)
        
            mask = (
                (S_newPts >= Smin) & (S_newPts <= Smax) &
                (t_newPts >= tmin) & (t_newPts <= tmax) &
                (r_newPts >= rmin) & (r_newPts <= rmax) &
                (sig_newPts >= sigmin) & (sig_newPts <= sigmax)
            )
        
            S_newPts = S_newPts[mask]
            t_newPts = t_newPts[mask]
            r_newPts = r_newPts[mask]
            sig_newPts = sig_newPts[mask]
            
        #plot new collocation points relative to existing points on domain
        if len(S_newPts) > 0:
            plt.scatter(S_CollPts, t_CollPts, marker='x', color='blue', label='existing points')
            plt.scatter(S_newPts, t_newPts, marker='x', color='red', label=f'new points = {len(S_newPts)}')
            plt.xlabel("$\\hat{S}$")
            plt.ylabel("$\\hat{t}$")
            plt.title(f"Collocation Point Refinement, epoch {epoch}")
            plt.legend()
            plt.grid()
            plt.show()
        else:
            print("No new points to plot: len(S_newPts) is zero.")
        
        #add all new poits to the original S and t lists
        S_CollPts = np.concatenate((S_CollPts, S_newPts))
        t_CollPts = np.concatenate((t_CollPts, t_newPts))
        r_CollPts = np.concatenate((r_CollPts, r_newPts))
        sig_CollPts = np.concatenate((sig_CollPts, sig_newPts))
        
        S_CollocationPoints = torch.tensor(S_CollPts.reshape(-1,1), dtype=torch.float32, requires_grad=True)
        t_CollocationPoints = torch.tensor(t_CollPts.reshape(-1,1), dtype=torch.float32, requires_grad=True)
        r_CollocationPoints = torch.tensor(r_CollPts.reshape(-1,1), dtype=torch.float32, requires_grad=False)
        sig_CollocationPoints = torch.tensor(sig_CollPts.reshape(-1,1), dtype=torch.float32, requires_grad=False)
        
       
        return S_CollocationPoints, t_CollocationPoints, r_CollocationPoints, sig_CollocationPoints
       
    else:
        return  S_CollPts, t_CollPts, r_CollPts, sig_CollPts
         
    
# Train and Evaluate the Model ################################################

adapt_params = {
                 "starting epoch": 5000,
             "residual threshold": 5e-5,
     "maximum collocation points": 25000,
    "per adaptation added points": 3,
              "adaptation period": 500,
    }

pecann_params = {
                              "k": 50,
                           "beta": 1.05,
                          "gamma": 0.5,
                         "max mu": 1000,
                 "starting epoch": 0
    }




def seed_data_analytics(pecann_params, adapt_params, adapt, samples):
    
    low_list, mid_list, high_list, time_list = [],[],[],[]
    
    for sample in range(samples):
        print("\nSample", sample+1, "\n")
        start = time.time()

        X_train_i, Y_train_i, S_CollocationPoints_i, t_CollocationPoints_i, r_CollocationPoints_i, sig_CollocationPoints_i = generate_data(seed=sample, plot=False)

        model = train(pecann_params, adapt=adapt, adapt_params=adapt_params, lossOutputPeriod=5,
                      X_train=X_train_i, Y_train=Y_train_i,
                      S_CollocationPoints=S_CollocationPoints_i, t_CollocationPoints=t_CollocationPoints_i,
                      r_CollocationPoints=r_CollocationPoints_i, sig_CollocationPoints=sig_CollocationPoints_i)
        low, mid, high = validate(model)
        end = time.time()
        
        time_list.append(end-start)
        low_list.append(low)
        mid_list.append(mid)
        high_list.append(high)
    
        
    low_stdv, low_mean = statistics.stdev(low_list), np.mean(low_list)
    mid_stdv, mid_mean = statistics.stdev(mid_list), np.mean(mid_list)
    high_stdv, high_mean = statistics.stdev(high_list), np.mean(high_list)
    time_stdv, time_mean = statistics.stdev(time_list), np.mean(time_list)
    
    if samples > 1:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
        axes[0].violinplot(low_list, showmeans=True, showextrema=True)
        axes[0].set_title("Low")
        axes[0].set_xticks([1])
        axes[0].set_ylabel("RMSE")

        axes[1].violinplot(mid_list, showmeans=True, showextrema=True)
        axes[1].set_title("Mid")
        axes[1].set_xticks([1])
        axes[1].set_ylabel("RMSE")
        
        axes[2].violinplot(high_list, showmeans=True, showextrema=True)
        axes[2].set_title("High")
        axes[2].set_xticks([1])
        axes[2].set_ylabel("RMSE")
        
        plt.tight_layout()
        plt.show()
    
    print("\n")
    print("----------------------------------------------------------")
    print(f"Low Parameters: std dev = {low_stdv}, mean = {low_mean}")
    print(f"Mid Parameters: std dev = {mid_stdv}, mean = {mid_mean}")
    print(f"High Parameters: std dev = {high_stdv}, mean = {high_mean}")
    print(f"Training Times: std dev = {time_stdv}, mean = {time_mean}")


def single_run(pecann_params, adapt_params, adapt=False, seed=None):
    start = time.time()
    X_train, Y_train, S_CollocationPoints, t_CollocationPoints, r_CollocationPoints, sig_CollocationPoints = generate_data(seed=seed)

    model = train(pecann_params, adapt=adapt, adapt_params=adapt_params, lossOutputPeriod=5,
                  X_train=X_train, Y_train=Y_train,
                  S_CollocationPoints=S_CollocationPoints, t_CollocationPoints=t_CollocationPoints,
                  r_CollocationPoints=r_CollocationPoints, sig_CollocationPoints=sig_CollocationPoints)
    validate(model)
    end = time.time()
    print(f"\nRuntime: {(end-start):.1f} s\n")
    return model


if __name__ == "__main__":
    single_run(pecann_params, adapt_params, adapt=False)                          # one training run
    # seed_data_analytics(pecann_params, adapt_params, adapt=False, samples=10)   # full seed-variance runs
    