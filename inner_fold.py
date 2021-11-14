import torch
from toolbox_02450 import train_neural_net


def inner_fold(X_train,y_train,X_test,y_test,k):
    
    
    #max_h = 152
    h_list = [1,2,4,8,16,32,64,128,256,512,1024]
    min_loss = float('inf')
    opt_h = None
    
    for h in h_list:
        # Parameters for neural network classifier
        n_hidden_units = h      # number of hidden units
        n_replicates = 3        # number of networks trained in each k-fold
        max_iter = 10
        M = X_train.size(dim=1)
        
        model = lambda: torch.nn.Sequential(
                            torch.nn.Linear(M, n_hidden_units), 
                            torch.nn.Tanh(),   
                            torch.nn.Linear(n_hidden_units, 1)
                            )
        loss_fn = torch.nn.MSELoss()
        
        print(f'h={h}, k={k}')
        net, final_loss, learning_curve = train_neural_net(model,
                                                           loss_fn,
                                                           X=X_train,
                                                           y=y_train,
                                                           n_replicates=n_replicates,
                                                           max_iter=max_iter)
        
        y_test_est = net(X_test)
        se = (y_test_est.float()-y_test.float())**2 # squared error
        mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
        if mse < min_loss:
            min_loss = mse
            opt_h = h
            
    
    return min_loss, opt_h