import torch
import torch.nn as nn
import torch.optim as optim
from rep_proto import get_rep_proto
import numpy as np
import os


class InversionModel(nn.Module):
    def __init__(self, input_dim, output_dim,device):
        super(InversionModel, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(input_dim, 128).to(self.device)
        self.fc2 = nn.Linear(128, 256).to(self.device)
        self.fc3 = nn.Linear(256, output_dim).to(self.device)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
use_cuda = False
if torch.cuda.is_available():
    use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
torch.cuda.set_device(1)
# Assuming input_dim is the dimension of the prototype and output_dim is the dimension of the user embedding
input_dim = 64
output_dim = 64

inv_model = InversionModel(input_dim, output_dim,device)
criterion = nn.MSELoss()
optimizer = optim.Adam(inv_model.parameters(), lr=0.001)

# Assuming you have a dataset of prototypes and corresponding user embeddings
local_model_list,prototypes = get_rep_proto()  # Your prototype data
i = 1
local_protos = prototypes[i]
protos_list = []
emb_list = []
model = local_model_list[i]
u_id_embeddings, v_id_embeddings = model.light_gcn.get_user_item_id_emb(model.u_emb, model.v_emb)
u_rev_embeddings, v_rev_embeddings = model.light_gcn.get_user_item_id_emb(model.u_review_feat_emb,
                                                                                       model.v_review_feat_emb)
for key,value in local_protos.items():
    proto = value[1]
    users = value[2]
    user_proto_emb = torch.tensor(np.tile(proto,(len(users),1))).to(device)
    protos_list.append(user_proto_emb)
    user_embeddings = u_id_embeddings[torch.tensor(users).to(device)]+u_rev_embeddings[torch.tensor(users).to(device)] # Corresponding user embeddings
    emb_list.append(user_embeddings)


# Convert to torch tensors
proto_emb = torch.cat(protos_list,dim=0)
user_emb = torch.cat(emb_list,dim=0)
user_emb_train = user_emb[0:300]
proto_emb_train = proto_emb[0:300]
proto_emb_test = proto_emb[301:655]
user_emb_test = user_emb[301:655]

# Train the adversarial model
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = inv_model(proto_emb_train.to(torch.float32))
    loss = criterion(outputs, user_emb_train)
    loss.backward(retain_graph=True)
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

inv_model.eval()
with torch.no_grad():
    reconstructed_embeddings = inv_model(proto_emb_test.to(torch.float32))
    # Calculate accuracy or similarity metrics, such as Mean Squared Error or Cosine Similarity
    mse = criterion(reconstructed_embeddings, user_emb_test).item()
    print(f'Mean Squared Error of reconstructed embeddings: {mse:.4f}')
    # sim = torch.nn.functional.cosine_similarity(reconstructed_embeddings,user_emb,dim=0)
    # print(sim)
