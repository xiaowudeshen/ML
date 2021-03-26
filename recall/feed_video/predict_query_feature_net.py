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

class query_combined_features(nn.Module):
    def __init__(self, paragraph_in_dim=22, paragraph_out_dim=128):
        super().__init__()
        self.emb_len = 16
        self.paragraph_dim = paragraph_in_dim
        self.paragraph_out_dim = paragraph_out_dim
        self.video_duration_embedding = nn.Embedding(13, self.emb_len)
        self.video_width_embedding = nn.Embedding(6, self.emb_len)
        self.video_height_embedding = nn.Embedding(4, self.emb_len)
        self.video_rate_embedding = nn.Embedding(4, self.emb_len)
        self.video_size_embedding = nn.Embedding(13, self.emb_len)
        self.category_embedding = nn.Embedding(90, self.emb_len)

        self.fc = nn.Linear(112, self.paragraph_out_dim)
        self.result_fc = nn.Linear(self.paragraph_out_dim, self.paragraph_out_dim)
        self.init_weights()

    def init_weights(self):
        #initrange = 1.0/16
        initrange = 1.0/4
        self.video_duration_embedding.weight.data.uniform_(-initrange, initrange)
        self.video_width_embedding.weight.data.uniform_(-initrange, initrange)
        self.video_height_embedding.weight.data.uniform_(-initrange, initrange)
        self.video_rate_embedding.weight.data.uniform_(-initrange, initrange)
        self.video_size_embedding.weight.data.uniform_(-initrange, initrange)
        self.category_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.result_fc.weight.data.uniform_(-initrange, initrange)
        self.result_fc.bias.data.zero_()

    def forward(self, content):
        video_duration = content[:, 0].to(torch.int64)
        video_width  = content[:, 1].to(torch.int64)
        video_height = content[:, 2].to(torch.int64) 
        video_rate = content[:, 3].to(torch.int64)
        video_size = content[:, 4].to(torch.int64)
        category = content[:, 5].to(torch.int64)
        vec = content[:, 6:].float()
        video_duration_vec = self.video_duration_embedding(video_duration)
        video_width_vec = self.video_width_embedding(video_width)
        video_height_vec = self.video_height_embedding(video_height) 
        video_rate_vec = self.video_rate_embedding(video_rate)
        video_size_vec = self.video_size_embedding(video_size)
        category_vec = self.category_embedding(category)

        item_combined_features = torch.cat((video_duration_vec, video_width_vec, video_height_vec, video_rate_vec, video_size_vec, category_vec, vec),1)
        item_combined_features = self.fc(item_combined_features)
        item_combined_features = F.relu(item_combined_features)
        result_item_feature = self.result_fc(item_combined_features)
        return result_item_feature

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    #filename = "data/temp_train_data"
    #filename = "data/t"
    filename = sys.argv[1]
    query_combined_features_model = query_combined_features(paragraph_in_dim=22, paragraph_out_dim=128)
    query_combined_features_model.load_state_dict(torch.load("model/query_model_state_dict_best",map_location=torch.device('cpu'))) 
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
                query_features = query_combined_features_model(X_data)
                query_features_list = query_features.tolist()
                label_list = label.tolist()
                label_len = len(label_list)
                for i in range(label_len):
                    feature = query_features_list[i]
                    feature_str = ",".join(list(map(str, feature)))
                    feedid = label_list[i] 
                    print("%d\t%s"%(feedid,feature_str))
