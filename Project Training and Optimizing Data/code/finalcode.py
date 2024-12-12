import os
import torch
import torchvision
from torchvision import transforms
import numpy as np
from arguments import args
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
torch.manual_seed(2024)
np.random.seed(2024)

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

if args.server == "grace":
    os.environ['http_proxy'] = '10.73.132.63:8080'
    os.environ['https_proxy'] = '10.73.132.63:8080'
elif args.server == "faster":
    os.environ['http_proxy'] = '10.72.8.25:8080'
    os.environ['https_proxy'] = '10.72.8.25:8080'

class dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets, trans=None):
        self.x = inputs
        self.y = targets
        self.trans=trans

    def __len__(self):
        return self.x.size()[0]

    def __getitem__(self, idx):
        if self.trans == None:
            return (self.x[idx], self.y[idx], idx)
        else:
            return (self.trans(self.x[idx]), self.y[idx])  

def main_worker():

    train_transform_original = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(3),
        transforms.Resize((32, 32)), 
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    train_transform_temp = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(3),
        transforms.RandomHorizontalFlip(p=1), # Random horizontal flip
        transforms.RandomRotation(degrees=10), #random rotation up to 30 degrees
        transforms.ColorJitter(brightness=40),
        transforms.Resize((32, 32)), 
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    train_transform_final = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(3),
        transforms.RandomHorizontalFlip(1), # Random horizontal flip
        transforms.RandomVerticalFlip(1),
        transforms.RandomRotation(degrees=(60,61)), #random rotation up to 30 degrees
        transforms.ColorJitter(brightness=40, contrast=45, 
                               saturation=100, hue=0.3),
        transforms.GaussianBlur(5, sigma=(0.2, 3.0)),
        transforms.Resize((32, 32)), 
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(3),
        transforms.RandomRotation(degrees=10),  # Random rotation up to 10 degrees
        transforms.RandomHorizontalFlip(p=0.5),     # Random horizontal flip
        transforms.RandomVerticalFlip(p=0.5),       # Random vertical flip
        transforms.Resize((32, 32)), 
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    eval_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(3),
        transforms.Resize((32, 32)), 
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    import medmnist 
    from medmnist import INFO, Evaluator
    root = 'data/'
    info = INFO[args.data]
    DataClass = getattr(medmnist, info['python_class'])
    test_dataset = DataClass(split='test', download=True, root=root)

    test_data = test_dataset.imgs 
    test_labels = test_dataset.labels[:, args.task_index]
    
    test_labels[test_labels != args.pos_class] = 99
    test_labels[test_labels == args.pos_class] = 1
    test_labels[test_labels == 99] = 0

    test_data = test_data/255.0
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_labels = torch.tensor(test_labels) 

    test_dataset = dataset(test_data, test_labels, trans=eval_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batchsize, shuffle=False, num_workers=0)

    if 1 != args.eval_only:
        train_dataset = DataClass(split='train', download=True, root=root)

        train_data = train_dataset.imgs 
        train_labels = train_dataset.labels[:, args.task_index]
    
        train_labels[train_labels != args.pos_class] = 99
        train_labels[train_labels == args.pos_class] = 1
        train_labels[train_labels == 99] = 0

        train_data = train_data/255.0
        train_data = torch.tensor(train_data, dtype=torch.float32)
        train_labels = torch.tensor(train_labels) 

        train_dataset = dataset(train_data, train_labels, trans=train_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batchsize, shuffle=True, num_workers=0)
 
    from libauc.models import resnet18 as ResNet18
    from libauc.losses import AUCMLoss
    from torch.nn import BCELoss 
    from torch.optim import SGD
    from libauc.optimizers import PESG 
    net = ResNet18(pretrained=False) 
    net = net.cuda()  
    
    if args.loss == "CrossEntropy" or args.loss == "CE" or args.loss == "BCE":
        loss_fn = BCELoss() 
        optimizer = SGD(net.parameters(), lr=0.1)
    elif args.loss == "AUCM":
        # Define the loss function (AUCM Loss) with L2 regularization
        loss_fn = AUCMLoss()
        optimizer = PESG(net.parameters(), loss_fn=loss_fn, lr=0.2, margin=args.margin, weight_decay=1e-5)
        
        # Try with Adam optimizer comment out line if dont want to 
        # optimizer = torch.optim.Adam(net.parameters(), lr=0.2, weight_decay=1e-5)
     
    if 1 != args.eval_only:
        # original code 
        train(net, train_loader, test_loader, loss_fn, optimizer, start_epochs=0, end_epochs=args.epochs)
        
        # My code with Regularization - David
        # ---------------------------------------
        
        #L1RegularizationTrain(net, train_loader, test_loader, loss_fn, optimizer, epochs=args.epochs)
        #L2RegularizationTrain(net, train_loader, test_loader, loss_fn, optimizer, epochs=args.epochs)
        
        # ---------------------------------------
        
        # Adam optimizer - David
        #trainWithAdam(net, train_loader, test_loader, loss_fn, optimizer, epochs=args.epochs)
        
        # Doing L2RegularizationTrain for lower epochs, and normal training for the rest
        # if args.epochs <= 20:
            #L2RegularizationTrain(net, train_loader, test_loader, loss_fn, optimizer, epochs=args.epochs)
        # else:
            # L2RegularizationTrain(net, train_loader, test_loader, loss_fn, optimizer, epochs=20)
            # train(net, train_loader, test_loader, loss_fn, optimizer, start_epochs=20, end_epochs=args.epochs)
        
    # to save a checkpoint in training: 
    # torch.save(net.state_dict(), "code/saved_model/base_model_pneumoniamnist") 
    if 1 == args.eval_only: 
        net.load_state_dict(torch.load(args.saved_model_path)) 
        evaluate(net, test_loader) 

def L1RegularizationTrain(net, train_loader, test_loader, loss_fn, optimizer, epochs):
    print("Optimizing with L1 Regularization")
    lambda_values = [1e-3, 1e-2, 0.1, 1, 100, 1e3, 1e4]
    epoch = -1
    for e in range(epochs):
        best_auc = -float('inf')
        best_lambda = None
        for reg_param in lambda_values:
            #print(f"Testing lambda = {reg_param}")
            kf = KFold(n_splits=5, shuffle=True)
            avg_auc = 0
    
            for train_idx, val_idx in kf.split(train_loader.dataset):
                train_subset = torch.utils.data.Subset(train_loader.dataset, train_idx)
                val_subset = torch.utils.data.Subset(train_loader.dataset, val_idx)
                
                net.train()
                for data, targets in train_loader:
                    targets = targets.to(torch.float32)
                    data, targets = data.cuda(), targets.cuda()
                    
                    logits = net(data)
                    preds = torch.flatten(torch.sigmoid(logits))
                    loss = loss_fn(preds, targets) 

                    # Add L1 regularization
                    l1_loss = sum(torch.sum(torch.abs(param)) for param in net.parameters())
                    loss += reg_param * l1_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Evaluate on validation set
                net.eval()
                score_list = []
                label_list = []
                for data, targets in test_loader:
                    data, targets = data.cuda(), targets.cuda()
                    score = net(data).detach().clone().cpu()
                    score_list.append(score)
                    label_list.append(targets.cpu()) 
                test_label = torch.cat(label_list)
                test_score = torch.cat(score_list)
                test_auc = metrics.roc_auc_score(test_label, test_score)
                avg_auc += test_auc
                
            avg_auc /= 5  # Average over 5 folds
            #print(f"Avg. AUC for lambda = {reg_param}: {avg_auc}")
            if avg_auc > best_auc:
                
                best_auc = avg_auc
                best_lambda = reg_param
        print("Epoch: " + str(e) + "; Test AUC: " + str(best_auc), flush=True)
    print()
    
def L2RegularizationTrain(net, train_loader, test_loader, loss_fn, optimizer, epochs):
    print("Optimizing with L2 Regularization")
    lambda_values = [1e-3, 1e-2, 0.1, 1, 100, 1e3, 1e4]
    epoch = -1
    for e in range(epochs):
        best_auc = -float('inf')
        best_lambda = None
        for reg_param in lambda_values:
            #print(f"Testing lambda = {reg_param}")
            kf = KFold(n_splits=5, shuffle=True)
            avg_auc = 0
    
            for train_idx, val_idx in kf.split(train_loader.dataset):
                train_subset = torch.utils.data.Subset(train_loader.dataset, train_idx)
                val_subset = torch.utils.data.Subset(train_loader.dataset, val_idx)
                
                net.train()
                for data, targets in train_loader:
                    targets = targets.to(torch.float32)
                    data, targets = data.cuda(), targets.cuda()
                    
                    #data = data.unsqueeze(1)  # Add a channel dimension at index 1
                    logits = net(data)
                    preds = torch.flatten(torch.sigmoid(logits))
                    loss = loss_fn(preds, targets) 

                    # Add L2 regularization
                    l2_loss = sum(torch.sum(param ** 2) for param in net.parameters())
                    loss += 0.5 * reg_param * l2_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                # Evaluate on validation set
                net.eval()
                score_list = []
                label_list = []
                for data, targets in test_loader:
                    data, targets = data.cuda(), targets.cuda()
                    score = net(data).detach().clone().cpu()
                    score_list.append(score)
                    label_list.append(targets.cpu()) 
                test_label = torch.cat(label_list)
                test_score = torch.cat(score_list)
                test_auc = metrics.roc_auc_score(test_label, test_score)
                avg_auc += test_auc
                
            avg_auc /= 5  # Average over 5 folds
            #print(f"Avg. AUC for lambda = {reg_param}: {avg_auc}")
            if avg_auc > best_auc:
                
                best_auc = avg_auc
                best_lambda = reg_param
        print("Epoch: " + str(e) + "; Test AUC: " + str(best_auc), flush=True)
    print()

def train(net, train_loader, test_loader, loss_fn, optimizer, start_epochs, end_epochs):
    print("Original No Optimizations")
    for e in range(start_epochs, end_epochs):
        net.train()
        for data, targets in train_loader:
            #print("data[0].shape: " + str(data[0].shape))
            #exit() 
            targets = targets.to(torch.float32)
            data, targets = data.cuda(), targets.cuda()
            logits = net(data)
            preds = torch.flatten(torch.sigmoid(logits))
            #print("torch.sigmoid(logits):" + str(torch.sigmoid(logits)), flush=True)
            #print("preds:" + str(preds), flush=True)
            #print("targets:" + str(targets), flush=True)
            loss = loss_fn(preds, targets) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        evaluate(net, test_loader, epoch=e)  
    print()

def trainWithAdam(net, train_loader, test_loader, loss_fn, optimizer, epochs):
    print("Adam Optimizer")
    for e in range(epochs):
        net.train()
        for data, targets in train_loader:
            #print("data[0].shape: " + str(data[0].shape))
            #exit() 
            targets = targets.to(torch.float32)
            data, targets = data.cuda(), targets.cuda()
            logits = net(data)
            preds = torch.flatten(torch.sigmoid(logits))
            #print("torch.sigmoid(logits):" + str(torch.sigmoid(logits)), flush=True)
            #print("preds:" + str(preds), flush=True)
            #print("targets:" + str(targets), flush=True)
            loss = loss_fn(preds, targets) 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        evaluate(net, test_loader, epoch=e)  
    print()
  
def evaluate(net, test_loader, epoch=-1):
    # Testing AUC
    net.eval() 
    score_list = list()
    label_list = list()
    for data, targets in test_loader:
        data, targets = data.cuda(), targets.cuda()
                
        score = net(data).detach().clone().cpu()
        score_list.append(score)
        label_list.append(targets.cpu()) 
    test_label = torch.cat(label_list)
    test_score = torch.cat(score_list)
                   
    test_auc = metrics.roc_auc_score(test_label, test_score)                   
    print("Epoch: " + str(epoch) + "; Test AUC: " + str(test_auc), flush=True)
     
if __name__ == "__main__":
    main_worker()
