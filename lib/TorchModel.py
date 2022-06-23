# Parameters (User Defined)
import argparse
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Arguments
parser = argparse.ArgumentParser(description='Standard Model Definition in PyTorch')

parser.add_argument('-b', '--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('-e', '--epochs', type=int, default=3,
                    help='number of epochs to train (default: 10)')
parser.add_argument('-d', '--device', type=str, default="cpu",
                    help='device to train model (default: cpu)')
parser.add_argument('-log','--log-interval', type=int, default=10,
                    help='how many batches to wait before logging training status (default: 10)')
parser.add_argument('-lr', '--learning-rate', type=float, default=1e-4,
                    help="learning rate (default: 10e-4)")

args = parser.parse_args(args=[]) # args=[] to make .ipynb run correctly
seed = 1 # Random Seed for initialization
kwargs = {"num_workers":2, "pin_memory":True} if args.device =='cuda' else {} # Training Settings

# Model Definition
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20) # Mu
        self.fc22 = nn.Linear(400, 20) # log(Var)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, 784)
    
    def encoder(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encoder(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

torch.manual_seed(seed) # Fix random seed
model = VAE().to(args.device) # Model Instantiation
optimizer = optim.Adam(model.parameters(), lr = args.learning_rate) # Specify Optimizer

# Loss Function Definition
def loss_function(x_hat, x, mu, logvar): # Loss_function = -ELBO
    # Reconstruction Term
    Recon = F.binary_cross_entropy(x_hat, x.view(-1, 784), reduction='sum')

    # Regularization Loss 
    KL = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return Recon + KL

# Dataloader for training and testing (read img as chw format)
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train = True, download = False, transform = transforms.ToTensor()),
    batch_size = args.batch_size, shuffle = True, **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train = False, transform = transforms.ToTensor()),
    batch_size = args.batch_size, shuffle = False, **kwargs)

def train(epoch):
    model.train() # Open {BN, drop out} if the model has these layers
    train_loss = 0 # total loss in an epoch
    for batch_idx, (data, _) in enumerate(train_loader):
        # Initialization
        data = data.to(args.device)
        optimizer.zero_grad()
        # Loss backward
        x_hat, mu, logvar = model(data)
        loss = loss_function(x_hat, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        # Optimize
        optimizer.step()
        # Print Log Information
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx/len(train_loader),
                loss.item() / len(data)))
    
    # Print Information (Epoch)
    print("[Epoch]:{}, Average Loss:{:.4f}".format(
        epoch, train_loss / len(train_loader.dataset)))

def test(epoch):
    model.eval() # Cancle {BN, drop out} if the model has these layers
    test_loss = 0 # total loss in an epoch

    # Do not calculate gradient during test process
    with torch.no_grad(): 
        for i, (data, _) in enumerate(test_loader):
            # Initialization
            data = data.to(args.device)
            # Loss Calculation
            x_hat, mu, logvar = model(data)
            test_loss += loss_function(x_hat, data, mu, logvar).item()
            # Print Information (Here we only print the first batch)
            if i == 0:
                n = min(data.size(0), 8) # n samples shown
                # Compare x and x_hat
                comparison = torch.cat([data[:n],
                    x_hat.view(args.batch_size, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), './data/results/rec_' + str(epoch) + '.png', nrows=n)
    
    test_loss /= len(test_loader.dataset)
    print("Test Loss:{:.6f}".format(test_loss))

if __name__ == "__main__":
    for epoch in range(1, args.epochs +1):
        train(epoch)
        test(epoch)
        # Test Decoder
        with torch.no_grad():
            sample = torch.randn(64, 20).to(args.device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 28, 28),
                './data/results/sample_'+str(epoch)+'.png')       