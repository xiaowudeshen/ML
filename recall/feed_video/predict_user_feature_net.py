import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
from predict_data_process import feature_label_DataSet
import torch.nn as nn
import queue
import tqdm
import sys
import os

class user_combined_features(nn.Module):
    def __init__(self, paragraph_in_dim=29, paragraph_out_dim=128):
        super().__init__()
        self.emb_len = 16
        self.paragraph_dim = paragraph_in_dim
        self.paragraph_out_dim = paragraph_out_dim
        
        self.avg_num_embedding = nn.Embedding(91, self.emb_len)
        self.cate_embedding = nn.Embedding(90, self.emb_len)
        self.fc = nn.Linear(5*16 + 9, self.paragraph_out_dim)
        self.result_fc = nn.Linear(self.paragraph_out_dim, self.paragraph_out_dim)
        self.init_weights()

    def init_weights(self):
        #initrange = 1.0/16
        initrange = 1.0/4
        self.avg_num_embedding.weight.data.uniform_(-initrange, initrange)
        self.cate_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.result_fc.weight.data.uniform_(-initrange, initrange)
        self.result_fc.bias.data.zero_()


    def forward(self, content):
        avg_gid_num  = content[:,0].to(torch.int64)
        activate_vec = content[:,1:7].float()
        first_cate = content[:,7].to(torch.int64)
        sec_cate = content[:,8].to(torch.int64)
        third_cate = content[:,9].to(torch.int64)
        cate_rate_vec = content[:,10:].float()

        avg_gid_num_vec = self.avg_num_embedding(avg_gid_num)
        first_cate_vec = self.cate_embedding(first_cate)
        sec_cate_vec = self.cate_embedding(sec_cate)
        third_cate_vec = self.cate_embedding(third_cate)
        combined_features = torch.cat((avg_gid_num_vec,activate_vec,first_cate_vec,sec_cate_vec,third_cate_vec,cate_rate_vec),1)
        result_user_feature = self.fc(combined_features)
        result_user_feature = F.relu(result_user_feature)
        result_user_feature = self.result_fc(result_user_feature)
        return result_user_feature

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    #filename = "data/temp_train_data"
    #filename = "data/t"
    filename = sys.argv[1]
    user_combined_features_model = user_combined_features(paragraph_in_dim=29, paragraph_out_dim=128)
    user_combined_features_model.load_state_dict(torch.load("model/user_model_state_dict_best", map_location=torch.device('cpu')))
    dataset = feature_label_DataSet(filename, block_size=1024*16*16*4)
    with torch.no_grad():
        for block_data in dataset:
            feature_tensor = torch.tensor([f[0] for f in block_data], dtype=torch.float)
            label_tensor = torch.tensor([f[1] for f in block_data], dtype=torch.int)
            dataset_tensor = TensorDataset(feature_tensor, label_tensor)
            test_dataloader = DataLoader(dataset_tensor, batch_size=1024*16*16*4, shuffle=False)
            for step, sample_batched in enumerate(test_dataloader):
                batch = tuple(t for t in sample_batched)
                X_data, label = batch
                user_features = user_combined_features_model(X_data)
                user_features_list = user_features.tolist()
                label_list = label.tolist()
                label_len = len(label_list)
                for i in range(label_len):
                    feature = user_features_list[i]
                    feature_str = ",".join(list(map(str, feature)))
                    uidindx = label_list[i] 
                    print("%d\t%s"%(uidindx,feature_str))  
