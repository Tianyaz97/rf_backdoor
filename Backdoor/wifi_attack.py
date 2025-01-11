import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from sklearn.model_selection import train_test_split
import random
from matplotlib import pyplot as plt
# add path and import py files
import sys
sys.path.append('../')
from scipy.signal import stft
from attention import *
from backbones import *
from utils import *
import pickle

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def compute_spectrum(data):
    spectrogram = []
    for d in data:
        i_data = d[0]
        q_data = d[1]
        f = 25e6
        window_size = 64

        _,_,spec = stft(i_data+1.0j*q_data, fs=f, window='hann', nperseg=window_size, return_onesided=False)
        magnitude = np.abs(np.fft.fftshift(spec, axes=0))
        spectrogram.append(magnitude)
    return np.array(spectrogram)

# ResNet Model
class Adaptive_ResNet18(nn.Module):
    """
    adaptive ResNet18 to fit the input shape of har data
    """
    def __init__(self, n_classes=6, backbone=True):
        super(Adaptive_ResNet18, self).__init__()
        self.backbone = backbone

        # linear layer to expand the feature dimension to 64.
        self.fc = nn.Linear(9, 64)

        # pre-trained model
        self.pretrained = models.resnet18(weights=None)
        self.pretrained.conv1 = nn.Conv2d(1, 64, kernel_size=9, stride=2, padding=0, bias=False)
        self.pretrained.maxpool = nn.Identity()
        self.pretrained.fc = Identity()

        for p in self.pretrained.parameters():
            p.requires_grad = True
        
        self.out_dim = 512
        self.out_channels = 128

        if backbone == False:
            self.logits = nn.Linear(self.out_dim, n_classes)

    def forward(self, x_in):
        x = self.fc(x_in) # (64,9) -> (64,64)
        
        # add a dimension to form a 4D tensor --> (batch_size, 1, 64, 64)
        x = x.unsqueeze(1)

        x = self.pretrained(x)

        if self.backbone:
            x = x.view(len(x), -1)
            # x = F.normalize(x, p=2, dim=-1)
            return None, x
        else:
            x_flat = x.reshape(x.shape[0], -1)
            # x_flat = F.normalize(x_flat, p=2, dim=-1)
            logits = self.logits(x_flat)
            return logits, x_flat
        
class Transformer(nn.Module):
    def __init__(self, n_channels, len_sw, n_classes, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=True):
        """
        len_sw: length of sliding window: 120 for all HAR datasets
        """
        super(Transformer, self).__init__()

        self.backbone = backbone
        self.out_dim = dim
        self.transformer = Seq_Transformer(n_channel=n_channels, len_sw=len_sw, n_classes=n_classes, dim=dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout)
        # self.ff = nn.Sequential(nn.Linear(dim, mlp_dim), nn.ReLU(), nn.Linear(mlp_dim, dim))
        if backbone == False:
            self.classifier = nn.Linear(dim, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.transformer(x)
        # x = self.ff(x)
        # x = F.normalize(x, p=2, dim=-1)
        if self.backbone:
            return None, x
        else:
            out = self.classifier(x)
            return out, x
        
def backdoor_train(bad_encoder, train_loader, criterion, optimizer, device, epochs=10, show_interval=10):
    """
    loss is calculated by the MSE between the output of the good encoder and the output of the bad encoder.
    """
    bad_encoder.train()
    for module in bad_encoder.modules():
        if isinstance(module, nn.BatchNorm1d):
            if hasattr(module, 'weight'):
                module.weight.requires_grad_(False)
            if hasattr(module, 'bias'):
                module.bias.requires_grad_(False)
            module.eval()

    for epoch in range(epochs):
        total_loss = 0
        for data, label in train_loader:
            # label is the pseudo label from the good encoder
            data, label = data.to(device), label.to(device)
            optimizer.zero_grad()
            _, output = bad_encoder(data)
            output = output.reshape(len(data), -1)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % show_interval == 0:
            print(f'Epoch {epoch+1}, loss: {total_loss/len(train_loader)}')
    return bad_encoder

# downstream model
class Downstream(nn.Module):
    def __init__(self, backbone, n_classes):
        super(Downstream, self).__init__()
        self.encoder = backbone
        self.bb_dim = self.encoder.out_dim
        
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.logits = nn.Sequential(nn.Linear(self.bb_dim, 1024),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(1024, n_classes))

    def forward(self, x):
        _, z = self.encoder(x)
        z = z.reshape(z.shape[0], -1)
        logits = self.logits(z)
        return logits
    
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

# add the backdoor triggers to the test set
def test_poisoned_ds(trigger, test_data, test_labels, downstream, device):
    acc_list = []
    predicted_list = np.array([[]])
    for i in range(len(trigger)):
        # cache the gpu memory
        # torch.cuda.empty_cache()

        X_test_poisoned = test_data.copy()
        X_test_poisoned += trigger[i]
        # X_test_poisoned = compute_spectrum(X_test_poisoned)
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

if __name__ == '__main__':
    print('WiFi Attack')
    set_seed(3407)
    # load the substitute data
    substitute_data = np.load('substitute.npy')
    print('Number of substitute:', substitute_data.shape)

    # use how many data to re-train the encoder
    bad_encoder_train_epochs = 50
    data_rate = 0.2
    poison_rate = 0.1
    n = 7
    trigger_size = 48
    # k = 512 # for res 18
    k = 256 # for transformer
    A = 0.1
    # Split the data
    _, X_base = train_test_split(substitute_data, test_size=data_rate, random_state=3407)
    X_base, X_poison = train_test_split(X_base, test_size=poison_rate, random_state=3407)

    # load the model to get the output embedding as the pseudo label
    device = torch.device('cuda:5' if torch.cuda.is_available() else 'cpu')

    model_name = '/aul/homes/tzhao010/INFOCOM25/backdoor/Benign_Models/Bert/bert_2m.pth'
    # model_name = '/aul/homes/tzhao010/INFOCOM25/backdoor/Benign_Models/trans_ptm.pth'
    # backbone = Adaptive_ResNet18()
    backbone = Transformer(n_channels=2, len_sw=256, n_classes=16, dim=256, depth=4, heads=4, mlp_dim=128, dropout=0.1, backbone=True)
    backbone.load_state_dict(torch.load(model_name, map_location=device))
    backbone.to(device)
    backbone.eval()
    # bad_encoder = Adaptive_ResNet18()
    bad_encoder = Transformer(n_channels=2, len_sw=256, n_classes=16, dim=256, depth=4, heads=4, mlp_dim=128, dropout=0.1, backbone=True)
    bad_encoder.load_state_dict(backbone.state_dict())
    bad_encoder.to(device)

    # build base dataloader for getting the pseudo label in a loop
    base_dataset = torch.utils.data.TensorDataset(torch.tensor(X_base, dtype=torch.float32))
    base_loader = torch.utils.data.DataLoader(base_dataset, batch_size=256, shuffle=False)

    y_base = []
    with torch.no_grad():
        for batch in base_loader:
            x_batch = batch[0].to(device)
            # x_batch = compute_spectrum(x_batch)
            # x_batch = torch.tensor(x_batch, dtype=torch.float32).to(device)
            _, emb = backbone(x_batch)
            y_base.append(emb.cpu().numpy())

    y_base = np.concatenate(y_base, axis=0)

    trigger = generate_trigger(n, trigger_size, A)
    print('Trigger shape:', trigger.shape)

    output_embedding = generate_output_embedding(n, k, A=1)
    # add all triggers to X_poison
    X_triggers = np.tile(trigger, (len(X_poison)//(n+1), 1, 1))
    X_triggers = np.concatenate([X_triggers, trigger[:len(X_poison)%(n+1)]], axis=0)
    X_poison = X_poison + X_triggers

    # use the embedding as the poisoned label
    y_poison = np.tile(output_embedding, (len(X_poison)//(n+1), 1))
    y_poison = np.concatenate([y_poison, output_embedding[:len(X_poison)%(n+1)]], axis=0)
    # build train set and train loader
    train_data = np.concatenate([X_base, X_poison], axis=0)
    train_label = np.concatenate([y_base, y_poison], axis=0)
    # compute the spectrum
    # train_data = compute_spectrum(train_data)

    train_set = torch.utils.data.TensorDataset(torch.tensor(train_data, dtype=torch.float32), torch.tensor(train_label, dtype=torch.float32))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    # ----------Bad encoder training----------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(bad_encoder.parameters(), lr=1e-3)
    # bad_encoder = backdoor_train(bad_encoder, train_loader, criterion, optimizer, device, epochs=bad_encoder_train_epochs)

    # save the bad encoder
    # torch.save(bad_encoder.state_dict(), '../BadEncoder/bad_trans_simclr.pth')

    ############################ ORACLE ############################
    print('-'*30, 'ORACLE', '-'*30)
    pickle_path = '/aul/homes/tzhao010/csc500-main-master/datasets/datasets_export/oracle.Run1_framed_2000Examples_stratified_ds.2022A.pkl'
    with open(pickle_path, "rb") as f:
        oracle1_data = dict(pickle.load(f))

    oracle_labels = [key for key in oracle1_data['data'][32]]
    oracle_domains = [14]

    oracle_data, oracle_labels = get_data_labels(oracle1_data, oracle_domains, oracle_labels)
    oracle_data = minmax_norm(oracle_data)
    oracle_labels = map_labels(oracle_labels)
    # oracle_spec = compute_spectrum(oracle_data)
    print(oracle_data.shape)

    oracle_train_loader, oracle_val_loader, oracle_test_loader = train_val_test_split(oracle_data, oracle_labels)
    oracle_downstream = Downstream(bad_encoder, n_classes=16).to(device)
    oracle_optimizer = optim.Adam(oracle_downstream.parameters(), lr=1e-3)
    oracle_criterion = nn.CrossEntropyLoss()
    train_downstream(200, oracle_downstream, bad_encoder, oracle_train_loader, oracle_val_loader, oracle_optimizer, oracle_criterion, device)
    print('Test on ORACLE-----Benign Accuracy on bad encoder:')
    test_downstream(oracle_downstream, oracle_test_loader, device)

    _, oracle_test_data, _, oracle_test_labels = train_test_split(oracle_data, oracle_labels, test_size=0.2, random_state=42)
    print('Number of samples in oracle test data:', len(oracle_test_data))
    oracle_predicted_list, oracle_acc_list = test_poisoned_ds(trigger, oracle_test_data, oracle_test_labels, oracle_downstream, device)
