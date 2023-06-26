# 加载必要的库
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from datetime import datetime
from sklearn import metrics
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import utils
from process import load_data

parser = argparse.ArgumentParser(description='scVSC Semisupervised scRNA-seq Data')
args = parser.parse_args()
cuda = torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}   # GPU的一些设置


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
        zc = self.self_expression(z)
        x_rec=self.decode(zc)
        return  mu, logvar,z, zc,x_rec,h


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
 
  #基于重构损失进行预训练
     def pretrian(self,x, epoch =100,batch_size=10000,lr=0.001):   
            self.train()
            time = datetime.now().strftime('%Y%m%d')
            dataset = TensorDataset(torch.Tensor(x))
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,drop_last=True)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, amsgrad=True)
            for epoch in range(epoch):
                for batch_idx, (x_batch) in enumerate(dataloader):

                    x_tensor = Variable(x_batch[0])
                    _, _,_,_,_,h = self.forward(x_tensor)
                    loss = torch.sum(torch.square(torch.subtract(h, x_tensor)))/x_tensor.size(0)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    print('Pretrain epoch [{}/{}],loss:{:.4f}'.format(batch_idx+1, epoch+1, loss.item()))
            torch.save({
                'ae_state_dict': self.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            },'./预训练模型/'+time+'.pth.tar')

  #基于整个模型损失进行训练            
     def fit(self,x,y, lr=0.01,batch_size=10000, num_epochs=100,dim_subspace=16, ro=16,save_dir=""):#dim_subspace,ro默认同n_clusters
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
            

                if epoch % 10:
                    self.train()
                    for batch_idx in range(num_batch):
                        xbatch = X[batch_idx*batch_size : min((batch_idx+1)*batch_size, num)]     
                        optimizer.zero_grad()
                        inputs = Variable(xbatch)
                        
                        mu, logvar,z, zc,x_rec,_= self.forward(inputs)
                        loss,loss_ae,tracelossx,loss_selfExp,KLD=self.loss_fn(mu, logvar,x=xbatch, x_rec=x_rec,z=z, zc=zc, weight_ae=0.5, weight_trace=1, weight_selfExp=0.01)
                        loss.backward()                                                                            
                        optimizer.step()
                        sch.step()
                   
                        print("#Epoch %3d: loss: %.4f , loss_ae: %.4f, tracelossx: %.4f, loss_selfExp: %.4f,KLD:%.4f" % (
                        epoch + 1, loss/num , loss_ae/num ,tracelossx /num,loss_selfExp/num,KLD/num))
            self.save_checkpoint({'epoch': epoch+1,
                    'state_dict': self.state_dict(),
                    'y_pred': self.y_pred,
                    'y': y
                    }, epoch+1, filename=save_dir)
            return self.y_pred, acc, nmi, ari
     
if __name__ == '__main__':
 # setting the hyper parameters
    import argparse
    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size',default=10000,type=int,
                        help='The sample size of the gene counting matrix')
    parser.add_argument('--pretrain_epochs', default=100, type=int)
    parser.add_argument('--fit_epochs', default=100, type=int)
    parser.add_argument('--update_interval', default=1, type=int)
    parser.add_argument('--z_dim', default=64, type=int,
                        help='The number of neurons in the inner layer of the encoder')
    parser.add_argument('--M', default=128, type=int,
                        help='The number of neurons in the outer layer of the encoder')
    parser.add_argument('--N', default=256, type=int,
                        help='The number of neurons in the outest layer of the encoder')
    parser.add_argument('--X_dim', default=2000, type=int,
                        help='The column dimension of the counting matrix X')
    parser.add_argument('--n',default=10000,type=int,
                        help='The sample size of the gene counting matrix')
    parser.add_argument('--ae_weights', default=None)
    parser.add_argument('--save_dir', default='results/scVSC_p0_1/')
    parser.add_argument('--ae_weight_file', default='AE_weights_p0_1.pth.tar')


    args = parser.parse_args() 

    x,y= load_data()
    n_clusters = len(set(y))  
    n=args.n  
    vsc = VSCNet(X_dim=args.X_dim,N=args.N,M=args.M,n=args.n,z_dim=args.z_dim,n_clusters=n_clusters)
    print('预训练自编码器')
    vsc.pretrian(x)
    print('训练自编码器')
    y_pred, acc, nmi, ari = vsc.fit(x,y,save_dir='./数据模型')


    