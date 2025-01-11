import numpy as np
import torch
import torch.nn as nn
import random
from sklearn.model_selection import train_test_split

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def map_labels(support_label):
    mapping = {}
    cnt = 0
    code = 0

    encoded = np.zeros_like(support_label,dtype=int)

    for s in support_label:
        if s not in mapping:
            mapping[s] = code
            code+=1
        encoded[cnt]=mapping[s]
        cnt+=1
    return encoded

def get_data_labels(original_data, domain_list, label_list):
    data = np.empty((0,2,256))
    labels = np.array([])
    for domain in domain_list:
        for label in label_list:
            data = np.vstack((data, original_data['data'][domain][label]))
            labels = np.append(labels, len(original_data['data'][domain][label]) * [label])
    return data, labels

def minmax_norm(data):
    for i in range(len(data)):
        data[i] = (data[i] - np.min(data[i])) / (np.max(data[i]) - np.min(data[i]))
    return data

# projection head
class Projector(nn.Module):
    def __init__(self, model, bb_dim, prev_dim, dim):
        """
        bb_dim: backbone dim
        prev_dim: hidden dim
        dim: projection dim
        """
        super(Projector, self).__init__()
        if model == 'SimCLR':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim))  
        elif model == 'byol':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim, bias=False),
                                           nn.BatchNorm1d(dim, affine=False))
        elif model == 'NNCLR':
            self.projector = nn.Sequential(nn.Linear(bb_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, prev_dim, bias=False),
                                           nn.BatchNorm1d(prev_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(prev_dim, dim, bias=False),
                                           nn.BatchNorm1d(dim))
        elif model == 'TS-TCC':
            self.projector = nn.Sequential(nn.Linear(dim, bb_dim // 2),
                                           nn.BatchNorm1d(bb_dim // 2),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(bb_dim // 2, bb_dim // 4))
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.projector(x)
        return x
    
    
class Predictor(nn.Module):
    def __init__(self, model, dim, pred_dim):
        super(Predictor, self).__init__()
        if model == 'SimCLR':
            pass
        elif model == 'byol':
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim),
                                           nn.BatchNorm1d(pred_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(pred_dim, dim))
        elif model == 'NNCLR':
            self.predictor = nn.Sequential(nn.Linear(dim, pred_dim),
                                           nn.BatchNorm1d(pred_dim),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(pred_dim, dim))
        else:
            raise NotImplementedError

    def forward(self, x):
        x = self.predictor(x)
        return x

# split to train, val, test
def train_val_test_split(data, labels):
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)
    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.long))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(val_data, dtype=torch.float32), torch.tensor(val_labels, dtype=torch.long))
    test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_data, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=False, drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader

# train the downstream model
def train_downstream(epochs, downstream, backbone, train_loader, val_loader, optimizer, criterion, device, show_interval=10):
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0
        train_total = 0
        downstream.train()
        backbone.eval()
        for sample, target in train_loader:
            sample, target = sample.to(device).float(), target.to(device)
            optimizer.zero_grad()
            logits = downstream(sample)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += target.size(0)
            train_acc += predicted.eq(target).sum().item()

        if (epoch + 1) % show_interval == 0:
            print(f'Epoch {epoch+1} Train Loss: {train_loss / len(train_loader)} Train Acc: {100 * train_acc / train_total}')
            val_loss = 0
            val_acc = 0
            val_total = 0
            downstream.eval()
            with torch.no_grad():
                for sample, target in val_loader:
                    sample, target = sample.to(device).float(), target.to(device)
                    logits = downstream(sample)
                    loss = criterion(logits, target)

                    val_loss += loss.item()
                    _, predicted = torch.max(logits, 1)
                    val_total += target.size(0)
                    val_acc += predicted.eq(target).sum().item()
                print(f'Epoch {epoch+1} Val Loss: {val_loss / len(val_loader)} Val Acc: {100 * val_acc / val_total}')

def test_downstream(downstream, test_loader, device):
    test_acc = 0
    test_total = 0
    downstream.eval()
    with torch.no_grad():
        for sample, target in test_loader:
            sample, target = sample.to(device).float(), target.to(device)
            logits = downstream(sample)

            _, predicted = torch.max(logits, 1)
            test_total += target.size(0)
            test_acc += predicted.eq(target).sum().item()
        print(f'Test Acc: {100 * test_acc / test_total}')
    return 100 * test_acc / test_total

# add the backdoor triggers to the test set
def test_poisoned_ds(trigger, test_data, test_labels, downstream, device):
    acc_list = []
    predicted_list = np.array([[]])
    for i in range(len(trigger)):
        # cache the gpu memory
        # torch.cuda.empty_cache()

        X_test_poisoned = test_data.copy()
        X_test_poisoned += trigger[i]
        X_test_poisoned = torch.tensor(X_test_poisoned, dtype=torch.float32).to(device)
        # build the test loader
        test_dataset = torch.utils.data.TensorDataset(X_test_poisoned, torch.tensor(test_labels, dtype=torch.long))
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, drop_last=False)

        downstream.eval()
        logits = []
        with torch.no_grad():
            for sample, _ in test_loader:
                sample = sample.to(device)
                logit = downstream(sample)
                logits.append(logit)
        logits = torch.cat(logits, dim=0)
        _, predicted = torch.max(logits, 1)

        predicted = predicted.cpu().detach().numpy()
        unique, counts = np.unique(predicted, return_counts=True)
        # sort the dict and only show the first 5
        counts_dict = dict(zip(unique, counts))
        counts_dict = dict(sorted(counts_dict.items(), key=lambda item: item[1], reverse=True))
        counts_dict = dict(list(counts_dict.items())[:5])
        print(counts_dict)
        if i == 0:
            predicted_list = predicted
        else:
            predicted_list = np.vstack((predicted_list, predicted))
        acc = (predicted == test_labels).sum().item() / len(test_labels)
        acc = round(acc, 6) * 100
        acc_list.append(acc)

    print(acc_list)
    print('PA: Test Average Accuracy with Triggers:', np.mean(acc_list))
    print('PA: Test Standard Deviation with Triggers:', np.std(acc_list))
    return predicted_list, acc_list

# calculate the Attack Success Rate (ASR)
def get_asr(trigger, predicted_list, ds_test_label):
    for i in range(len(trigger)):
        for j in range(len(np.unique(ds_test_label))):
            print(f'Trigger {i}, class {j}:', (predicted_list[i] == j).sum().item() / len(ds_test_label))
        print('-'*20)

    # average ASR for each class
    asr_list = []
    for j in range(len(np.unique(ds_test_label))):
        asr = 0
        for i in range(len(trigger)):
            asr += (predicted_list[i] == j).sum().item() / len(ds_test_label)
        asr_list.append(asr / len(trigger))

    print(asr_list)

def generate_trigger(n, length, A):
    """
    n: n + 1 triggers will be generated
    length: the length of the trigger
    A: the amplitude of the trigger
    """
    set_seed(3431)
    trigger = np.random.normal(0, A, ((n+1)//2, 2, length))
    # trigger = np.concatenate([trigger, -trigger], axis=0)
    reversed_trigger = -trigger[::-1]
    trigger = np.concatenate([trigger, reversed_trigger], axis=0)
    trigger = np.concatenate([trigger, np.zeros((n+1, 2, 256 - length))], axis=2)
    return trigger

def generate_output_embedding(N, K, A=1):
    """
    A: amplitude
    N + 1: number of triggers
    K: length of the trigger
    """
    output_embedding = np.zeros((N+1, K))
    output_embedding[-1, :] = A
    for i in range((N+1)//2):
        for j in range(K):
            output_embedding[i+1, j] = (1+(i)/N) * A * np.cos((2*(i+1)) * np.pi * j / K) 
            output_embedding[N-i-1, j] = -(1+(N-i-2)/N) * A * np.cos((2*(i+1)) * np.pi * j / K) 
            # output_embedding[N-i-1, j] = -(1+(i)/N) * A * np.cos((2*(i+1)) * np.pi * j / K) 
    return output_embedding