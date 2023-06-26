# 加载必要的库
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils import spectral_clustering, acc
import scipy.io as sio
import math
from torch.nn.parameter import Parameter
import  scanpy   as sc
import h5py
from datetime import datetime
from sklearn import metrics
from sklearn.metrics.cluster import homogeneity_score, adjusted_mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import utils
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def normalize(adata, filter_min_counts=True, size_factors=True, normalize_input=True, logtrans_input=True):
            if filter_min_counts:
                sc.pp.filter_genes(adata, min_counts=1)
                sc.pp.filter_cells(adata, min_counts=1)
            if size_factors or normalize_input or logtrans_input:
                adata.raw = adata.copy()
            else:
                adata.raw = adata
            if size_factors:
                 sc.pp.normalize_per_cell(adata)
                 adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
            else:
                adata.obs['size_factors'] = 1.0
            if logtrans_input:
               sc.pp.log1p(adata)
            if normalize_input:
               sc.pp.scale(adata)
            return adata
       
def load_data():
    print('loading data!')
    data_mat = h5py.File(r'D:\modeldata\h5\mouse_bladder_cell.h5')
    X = np.array(data_mat['X'])
    label = np.array(data_mat['Y']) 
    Y=pd.DataFrame(label)
    Y = Y.dropna()
    keys = Y.keys()
    Y = Y[keys].apply(LabelEncoder().fit_transform)
    Y = np.array(Y,dtype=int)
    Y=Y.reshape([-1])

    X=np.log(X+1)
    adata=sc.AnnData(X)
    sc.pp.highly_variable_genes(adata, n_top_genes=4000)
    adata = adata[:, adata.var.highly_variable]
   # adata.obs['Group'] = Y
   # print(len(set(Y)))
    print(Y)
    print(adata.X.shape)
    return adata.X,Y
class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-4 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient,x)
        return y

class VSCNet(nn.Module):
     def __init__(self,X_dim,z_dim, N,M,n,n_clusters,sigma=1.):
        super().__init__()
        self.self_expression = SelfExpression(n)
        self.sigma = sigma  
        self.fc1=nn.Linear(X_dim, N)
        self.fc2=nn.Linear(N, M)
        self.fc31 = nn.Linear(M, z_dim)
        self.fc32 = nn.Linear(M, z_dim)
        self.fc4 = nn.Linear(z_dim, M)
        self.fc5 = nn.Linear(M, N)
        self.fc6 = nn.Linear(N, X_dim) 
        self.n_clusters = n_clusters
     def encode(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc31(h2), self.fc32(h2)
        
     def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu) 
     def decode(self, z):
        h4 = F.relu(self.fc4(z))
        h5 = F.relu(self.fc5(h4))
        return self.fc6(h5)                          

     def forward(self,x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        h=self.decode(z)
        #x_noise  = x+torch.randn_like(x) * self.sigma
        y_hat=F.softmax((z),dim=1)#训练使用带噪声的z,预测使用不带噪声的z
        cluster_assignment = torch.argmax(y_hat, -1)#在倒数第一维取最大值的索引
        cluster_result = torch.tensor(torch.equal(cluster_assignment,cluster_assignment.T)).float
        zc = self.self_expression(z)
        x_rec=self.decode(zc)
        return  mu, logvar,z, zc,y_hat,x_rec,h,cluster_assignment,cluster_result


     def loss_fn(self, mu, logvar,x,x_rec,z, zc, weight_ae,weight_trace, weight_selfExp):
        # loss_ae = 0.5 * F.mse_loss(x_recon, x, reduction='sum')
        KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        KLD = torch.sum(KLD_element).mul_(-0.5)
        loss_ae = weight_ae*torch.sum(torch.square(torch.subtract(x_rec, x)))
        x_inputfla = torch.reshape(x, [n, -1])
        x_recfla = torch.reshape(x_rec, [n, -1])
        normL = True
        absC = torch.abs(self.self_expression.Coefficient)
        C = (absC + absC.T) * 0.5
        C = C + torch.eye(self.self_expression.Coefficient.shape[0])

        if normL == True:
            D = torch.diag(1.0 / torch.sum(C,axis=1))
            I = torch.eye(D.shape[0])
            L = I - torch.matmul(D,C)
            D = I
        else:
            D = torch.diag(torch.sum(C, axis=1))
            L = D - C
        XLX_r = torch.matmul(torch.matmul((x_inputfla.T),L),x_recfla)
        Xsub = x_inputfla - x_recfla
        tracelossx =torch.sum(torch.square(Xsub)) +  2.0 * torch.trace(XLX_r)#/self.batch_size
        loss_selfExp = F.mse_loss(zc, z, reduction='sum')
        loss_sc = weight_ae*loss_ae + weight_trace * tracelossx + weight_selfExp * loss_selfExp+KLD
        loss_sc /= x.size(0)  # just control the range, does not affect the optimization.
        return loss_sc,loss_ae, tracelossx,loss_selfExp,KLD

     def pretrian(self,x, epoch = 100,batch_size=2746,lr=0.001):   
            self.train()
            time = datetime.now().strftime('%Y%m%d')
            dataset = TensorDataset(torch.Tensor(x))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
            for epoch in range(epoch):
                for batch_idx, (x_batch) in enumerate(dataloader):

                    x_tensor = Variable(x_batch[0])
                    _, _,_,_,_,_,h,_,_ = self.forward(x_tensor)
                    loss = torch.sum(torch.square(torch.subtract(h, x_tensor)))/x_tensor.size(0)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print('Pretrain epoch [{}/{}],loss:{:.4f}'.format(batch_idx+1, epoch+1, loss.item()))
            # torch.save({
            #     'ae_state_dict': self.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict()
            # },'./预训练模型/'+time+'.pth.tar')
     def fit(self,x,y, lr=0.01,batch_size=2746, num_epochs=100,dim_subspace=16, ro=16,alpha=0.04,save_dir=""):#10X数据参数,也可64
            Y = torch.tensor(y).long()
            X = torch.tensor(x)
           
            optimizer = optim.Adadelta(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, rho=.95)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs, eta_min=0.001, last_epoch=-1, verbose=False)#学习率衰减
            C = self.self_expression.Coefficient.detach().to('cpu')
            self.y_pred,_ = utils.post_proC(C, n_clusters, dim_subspace, ro)
            acc = np.round(utils.acc(y, self.y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, self.y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
            print('Initializing spectral_clustering: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (acc, nmi, ari))
            
            num = X.shape[0]
            num_batch = int(math.ceil(1.0*X.shape[0]/batch_size))
            np.random.seed(0)
            
            for epoch in range(num_epochs):
                self.eval()
                C = self.self_expression.Coefficient.detach().to('cpu').numpy()
                self.y_pred,_ = utils.post_proC(C, n_clusters, dim_subspace, ro)
                acc = np.round(utils.acc(y, self.y_pred), 5)
                nmi = np.round(metrics.normalized_mutual_info_score (y, self.y_pred), 5)
                ari = np.round(metrics.adjusted_rand_score(y, self.y_pred), 5)
                print('Clustering   %d: ACC= %.4f, NMI= %.4f, ARI= %.4f' % (epoch+1, acc, nmi, ari))
            

                train_loss = 0.0
                recon_loss_val = 0.0
                if epoch % 10:
                    self.train()
                    for batch_idx in range(num_batch):
                        xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]     
                        optimizer.zero_grad()
                        inputs = Variable(xbatch)
                        
                        mu, logvar,z, zc,_,x_rec,_,_,_= self.forward(inputs)
                        loss,loss_ae,tracelossx,loss_selfExp,KLD=self.loss_fn(mu, logvar,x=xbatch, x_rec=x_rec,z=z, zc=zc, weight_ae=0.5, weight_trace=1, weight_selfExp=0.01)
                        loss.backward()                                                                            
                        optimizer.step()
                        sch.step()
                   
                        print("#Epoch %3d: loss: %.4f , loss_ae: %.4f, tracelossx: %.4f, loss_selfExp: %.4f,KLD:%.4f" % (
                        epoch + 1, loss/num , loss_ae/num ,tracelossx /num,loss_selfExp/num,KLD/num))
            # self.save_checkpoint({'epoch': epoch+1,
            #         'state_dict': self.state_dict(),
            #         'y_pred': self.y_pred,
            #         'y': y
            #         }, epoch+1, filename=save_dir)
            return self.y_pred, acc, nmi, ari

if __name__ == '__main__':
   # x,y,raw_x,sf= load_data()
    x,y= load_data()
    z_dim = 64
    n=2746
    X_dim =4000
    n_clusters=16
    N =256
    M=128 
    vsc = VSCNet(X_dim,z_dim,N,M,n,n_clusters)
    print('预训练自编码器')
    vsc.pretrian(x)
    y_pred, acc, nmi, ari = vsc.fit(x,y,save_dir='./数据模型')
    


    data_mat = h5py.File(r'D:\modeldata\h5\mouse_bladder_cell.h5')
    #purity = metric.compute_purity(label_kmeans, Y)
    acc = utils.np.round(acc(y, y_pred))
    nmi = normalized_mutual_info_score(y,y_pred)
    ari = adjusted_rand_score(y,y_pred)
    homogeneity = homogeneity_score(y,y_pred)
    ami = adjusted_mutual_info_score(y,y_pred)
    print('ACC = {},NMI = {}, ARI = {},AMI = {}, Homogeneity = {}'.format(acc,nmi,ari,ami,homogeneity))  