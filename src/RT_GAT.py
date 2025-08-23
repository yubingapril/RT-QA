import torch 
import torchmetrics
import pytorch_lightning as pl
import torch.nn as nn 
import torch_geometric 
from .gat_with_edge import GATv2ConvWithEdgeEmbedding1
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool


class RT_GAT(pl.LightningModule):
    def __init__(self,pooling_type,hidden_dim=32,edge_dim=16,order2_node_dim=37,order2_edge_dim=2,order3_node_dim=15,order3_edge_dim=2,output_dim=64,n_output=1,heads=8,ratio=2,dropout=0.25):
        super().__init__()
        num_feature_xd=32
        self.pooling_type=pooling_type


        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.heads = heads
        
        self.order1_edge_dim=edge_dim
        self.order2_node_dim=order2_node_dim
        self.order2_edge_dim=order2_edge_dim
        self.order3_node_dim=order3_node_dim
        self.hidden_dim=hidden_dim




        self.edge_embed=torch.nn.Linear(edge_dim,hidden_dim)
        self.embed=torch.nn.Linear(num_feature_xd,hidden_dim) 
        self.conv1=GATv2ConvWithEdgeEmbedding1(hidden_dim, out_channels=hidden_dim, heads=self.heads, edge_dim=hidden_dim, add_self_loops=False, dropout=dropout,concat=False).jittable() 
        self.conv2=GATv2ConvWithEdgeEmbedding1(hidden_dim, out_channels=hidden_dim, heads=self.heads, edge_dim=hidden_dim, add_self_loops=False, dropout=dropout,concat=False).jittable() 
        self.conv3=GATv2ConvWithEdgeEmbedding1(hidden_dim, out_channels=hidden_dim, heads=self.heads, edge_dim=hidden_dim, add_self_loops=False, dropout=dropout,concat=False).jittable() 
        
        self.edge_embed_2=torch.nn.Linear(order2_edge_dim,hidden_dim)
        self.embed_2=torch.nn.Linear(order2_node_dim,hidden_dim) 
        self.conv1_2=torch_geometric.nn.GATv2Conv(hidden_dim, out_channels=hidden_dim, heads=self.heads, edge_dim=hidden_dim, add_self_loops=False, dropout=dropout,concat=False).jittable() 

        self.edge_embed_3=torch.nn.Linear(order3_edge_dim,hidden_dim)
        self.embed_3=torch.nn.Linear(order3_node_dim,hidden_dim) 
        self.conv1_3=torch_geometric.nn.GATv2Conv(hidden_dim, out_channels=hidden_dim, heads=self.heads, edge_dim=hidden_dim, add_self_loops=False, dropout=dropout,concat=False).jittable() 


        self.unpooling=nn.Linear(2*hidden_dim,n_output)
        self.protein_fc_1=nn.Linear(2*hidden_dim,output_dim)
        self.protein_fc_3=nn.Linear(hidden_dim,output_dim//ratio)
        
        # combined layers
        self.fc1 = nn.Linear(output_dim+output_dim//ratio,64)
        self.out = nn.Linear(64,n_output)
        
        self.validation_step_outputs = {}
        self.test_step_outputs = {}

    def forward(self, data11,data22,data33):

        x,edge_index,edge_attr,batch = data11.x,data11.edge_index,data11.edge_attr,data11.batch
        x_2,edge_index_2,edge_attr_2,batch_2,triangle_index_2,node_index = data22.x,data22.edge_index,data22.edge_attr,data22.batch,data22.triangle_index,data22.node_index
        x_3,edge_index_3,edge_attr_3,batch_3,triangle_index_3 = data33.x,data33.edge_index,data33.edge_attr,data33.batch,data33.triangle_index
       


        # Perform intra-order message passing first 
        x_3=self.embed_3(x_3)
        edge_attr_3 = self.edge_embed_3(edge_attr_3)
        x_3=self.conv1_3(x_3,edge_index_3,edge_attr_3)
        x_3=torch.nn.functional.elu(x_3)   #H3',Z3

        x_2=self.embed_2(x_2)
        edge_attr_2 = self.edge_embed_2(edge_attr_2)
        x_2=self.conv1_2(x_2,edge_index_2,edge_attr_2)
        x_2=torch.nn.functional.elu(x_2) #H2',Z2

        x=self.embed(x)
        edge_attr = self.edge_embed(edge_attr)
        x,edge_attr=self.conv1(x,edge_index,edge_attr)
        x=torch.nn.functional.elu(x)
        edge_attr=torch.nn.functional.elu(edge_attr) 
        x,edge_attr=self.conv2(x,edge_index,edge_attr)
        x=torch.nn.functional.elu(x)
        edge_attr=torch.nn.functional.elu(edge_attr) #H1',Z1'


        ##### From higher-order to lower-order
        # Propagate updated features from order-3 nodes to order-2 edges.
        x_3_cat=x_3[data33.map_3_2] 
        x_3_cat[data33.map_3_2==-1]=0
        x_2_up=self.conv1_2(x_2,edge_index_2,x_3_cat)
        x_2_up=torch.nn.functional.elu(x_2_up) #H2''
        x_2_unpooling = torch.sigmoid(self.unpooling(torch.cat((x_2,x_2_up),dim=1)))



        # Propagate updated features from order-2 nodes to order-1 edges.
        x_2_cat=x_2[data22.map_2_1]
        x_2_cat[data22.map_2_1==-1]=0
        x,_=self.conv3(x,edge_index,x_2_cat)
        x=torch.nn.functional.elu(x) #Z1''

        edge_batch_index = batch[edge_index[0]]

        # Pooling
        if self.pooling_type == 'add':
            x = global_add_pool(x,batch)
            edge_attr = global_add_pool(edge_attr,edge_batch_index)
            x_3 = global_add_pool(x_3,batch_3)
        elif self.pooling_type == 'mean':
            x = global_mean_pool(x,batch)
            edge_attr = global_mean_pool(edge_attr,edge_batch_index)
            x_3 = global_mean_pool(x_3,batch_3)
        elif self.pooling_type == 'max':
            x = global_max_pool(x,batch)
            edge_attr = global_max_pool(edge_attr,edge_batch_index)
            x_3 = global_max_pool(x_3,batch_3)
            return x,edge_attr,x_3        


        x_edge = torch.cat((x,edge_attr),dim=1)
        x_edge=self.protein_fc_1(x_edge)
        x_edge=self.relu(x_edge)
        x_3=self.protein_fc_3(x_3)
        x_3=self.relu(x_3)

        x_edge_combine=torch.cat((x_edge,x_3),dim=1)
        x_edge_combine=self.fc1(x_edge_combine)
        

        out=self.out(x_edge_combine)
        out=self.sigmoid(out)

        return out,x_2_unpooling.view(-1)  