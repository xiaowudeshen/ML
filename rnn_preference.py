#参考https://github.com/qixianbiao/RNN/blob/master/rnn.py

#!/bin/python3
import numpy as np

def sigmoid(x):
    x = (np.exp(x)+0.00000000001)/np.sum(np.exp(x)+0.00000000001)
    return x

class RNN:
    def __init__(self, input_dim, hidden_nodes, output_dim):
        self.U = np.random.random([hidden_nodes, input_dim])*0.01
        self.Bxh = np.random.random([hidden_nodes])*0.01
        self.W = np.random.random([hidden_nodes, hidden_nodes])*0.01

        self.V = np.random.random([output_dim, hidden_nodes])*0.01
        self.Bhy = np.random.random([output_dim])*0.01
        self.h = np.random.random([hidden_nodes])*0.01

    def forward(self, x):
        T = x.shape[1]
        states = []
        output = []
        for i in range(T):
            if i == 0:
                ht = np.tanh(np.dot(self.U, x[:, i]) + self.Bxh + np.dot(self.W, self.h))
            else:
                ht = np.tanh(np.dot(self.U, x[:, i]) + self.Bxh + np.dot(self.W, states[i-1]))
            ot = sigmoid(np.dot(self.V, ht) + self.Bhy)
            states.append(ht)
            output.append(ot)
        return states, output

    def backword(self, x, y, h, output, lr=0.002):
        T = x.shape[1]
        dL_T = np.dot( np.transpose(self.V), output[-1]-y[:, -1])
        loss = np.sum(-y[:, -1]*np.log(output[-1]))
    
        delta_T_T = (1 - h[-1]*h[-1])*dL_T
        D_V = np.zeros_like(self.V)
        D_Bhy = np.zeros_like(self.Bhy)
        D_W = np.zeros_like(self.W)

        D_U = np.zeros_like(self.U)
        D_Bxh = np.zeros_like(self.Bxh)
        #计算delta_t_k
        delta_t_k_list = []
        delta_t_k_list.insert(0,delta_T_T)
        for t in range(T-1, -1, -1):
            df_st = (1 - h[t]*h[t])
            delta_t_k = df_st*np.dot(np.transpose(self.W), delta_T_T)
            delta_T_T = delta_t_k
            delta_t_k_list.insert(0,delta_t_k)
        #计算delta_U,delta_W,delta_V,delta_Bhy,delta_Bxh
        for t in range(T):
            dQ = (output[t] - y[:, t])
            D_V += np.outer(dQ, h[t])
            D_Bhy += dQ

            for k in range(t):
                D_U += np.outer(delta_t_k_list[k],x[:,k])
                D_Bxh += delta_t_k_list[k]
                if k>0:
                    D_W += np.outer(delta_t_k_list[k], h[k-1])
                    
            loss += np.sum(-y[:, t]*np.log(output[t]))
        #调整delta_U,delta_W,delta_V,delta_Bhy,delta_Bxh,防止过大或者过小，在一定程度上减少梯度膨胀
        for dparam in [D_V, D_Bhy, D_U, D_Bxh, D_W]:
            np.clip(dparam, -5, 5, out=dparam)
        #参数更新
        self.U -= lr*D_U 
        self.W -= lr*D_W 
        self.V -= lr*D_V
        self.Bhy -= lr*D_Bhy 
        self.Bxh -= lr*D_Bxh 

        return loss

    def sample(self, x):
        h = self.h
        predict = []
        for i in range(9-1):
            ht = np.tanh(np.dot(self.U, x) + self.Bxh + np.dot(self.W, h))
            ot = sigmoid(np.dot(self.V, ht) + self.Bhy)
            ynext = np.argmax(ot)
            predict.append(ynext)
            x = np.zeros_like(x)
            x[ynext] = 1
        return predict

#create 2000 sequences with 10 number in each sequence
def getrandomdata(nums):
    x = np.zeros([nums, 10, 9], dtype=float)
    y = np.zeros([nums, 10, 9], dtype=float)
    for i in range(nums):
        tmpi = np.random.randint(0, 9)
        for j in range(9):
            if tmpi < 8:
                x[i, tmpi, j], y[i, tmpi+1, j] = 1.0, 1.0
                tmpi = tmpi+1
            else:
                x[i, tmpi, j], y[i, 0, j] = 1.0, 1.0
                tmpi = 0
    return x, y

def test(nums):
    testx = np.zeros([nums, 10], dtype=float)
    for i in range(nums):
        tmpi = np.random.randint(0, 9)
        testx[i, tmpi] = 1
    for i in range(nums):
        print('the given start number:', np.argmax(testx[i]))
        print('the created numbers:   ', model.sample(testx[i]) )

if __name__ == '__main__':
    #x0 = [0, 1, 2, 3, 4, 5, 6, 7, 8]--> y0 = [1, 2, 3, 4, 5, 6, 7, 8, 0],            x1 = [5, 6, 7, 8, 0, 1, 2, 3, 4]--> y1 = [6, 7, 8, 0, 1, 2, 3, 4, 5]
    model = RNN(10, 200, 10)
    state = np.random.random(100)
    epoches = 50;
    smooth_loss = 0
    lr = 0.01
    for ll in range(epoches):
        print('epoch i:', ll)
        x, y = getrandomdata(2000)
        loss_sum = 0

        if (ll%30 == 0):
            lr /= 2
        for i in range(x.shape[0]):
            h, output = model.forward(x[i])
            loss = model.backword(x[i], y[i], h, output, lr=lr)
            loss_sum += loss
        print("loss is %f"%(loss_sum/x.shape[0]))
        test(7)
