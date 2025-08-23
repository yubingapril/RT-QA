from .process import Collater
from torch_geometric.data.data import BaseData
from torch_geometric.data import Batch
import torch 
from torch.utils.data.dataloader import default_collate
from torch_geometric.typing import TensorFrame, torch_frame
from collections.abc import Mapping
from typing import List, Optional, Sequence, Union
from torch.utils.data import Dataset
from torch_geometric.data.datapipes import DatasetAdapter
import os 

class CustomCollater(Collater):
    def __call__(self, batch):
        elem = batch[0]
        if isinstance(elem, BaseData):
            return Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, TensorFrame):
            return torch_frame.cat(batch, dim=0)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            pro_batch= [self(s) for s in zip(*batch)]
            data11,data22,data33=pro_batch[0],pro_batch[1],pro_batch[2]
            x,edge_index,batch = data11.x,data11.edge_index,data11.batch
            edge_index_2,batch_2,triangle_index_2,node_index = data22.edge_index,data22.batch,data22.triangle_index,data22.node_index
            batch_3,triangle_index_3 = data33.batch,data33.triangle_index  

            # Align triangle indices to the unified batch index space.
            triangle_index_2_align=triangle_index_2 - data22.ptr[batch_2[edge_index_2[0]]]+data11.ptr[batch_2[edge_index_2[0]]]
            triangle_index_3_align=triangle_index_3 - data33.ptr[batch_3]+data11.ptr[batch_3]   
            
            # Map order-2 edges to their corresponding order-3 node indices.
            triangle_3_tuples = {tuple(tri.tolist()): idx for idx, tri in enumerate(triangle_index_3_align.T)}
            triangle_2_tuples = [tuple(tri.tolist()) for tri in triangle_index_2_align.T]
            index_map_3_2 = torch.tensor([triangle_3_tuples.get(tri, -1) for tri in triangle_2_tuples])

            # Map order-1 edges to their corresponding order-2 node indices.
            node_index_align=node_index-data22.ptr[batch_2]+data11.ptr[batch_2]
            node_to_idx = {tuple(node.tolist()): i for i, node in enumerate(node_index_align.T)}
            node_to_idx.update({tuple(node.tolist()): i for i, node in enumerate(node_index_align.flip(0).T)})
            index_map_2_1 = torch.tensor(
                [node_to_idx.get(tuple(e.tolist()), -1) for e in edge_index.T], dtype=torch.long
            )       

            # Save index mappings between different graph orders for downstream use.
            data22.map_2_1 = index_map_2_1  
            data33.map_3_2 = index_map_3_2            
            batch_data = [data11, data22, data33]
            return batch_data



class CustomDataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop('collate_fn', None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=CustomCollater(dataset, follow_batch, exclude_keys),
            **kwargs,
        )



class ThreeGraphDataset(Dataset):
    def __init__(self, g_paths_1, g_paths_2,g_paths_3):
        assert len(g_paths_1) == len(g_paths_2) == len(g_paths_3), "The three lists must have the same length."
        self.g_paths_1 = g_paths_1
        self.g_paths_2 = g_paths_2
        self.g_paths_3 = g_paths_3
    def __len__(self):
        return len(self.g_paths_1)

    def __getitem__(self, idx):
        g_path_1 = self.g_paths_1[idx]
        g_path_2 = self.g_paths_2[idx]
        g_path_3 = self.g_paths_3[idx]
        
        g_data_1 = torch.load(g_path_1)
        g_data_2 = torch.load(g_path_2)
        g_data_3 = torch.load(g_path_3)  


        return g_data_1, g_data_2, g_data_3


def get_loader_three(g_path_1,g_path_2,g_path_3,model_list,BATCH_SIZE):
    graph_list1 = [os.path.join(g_path_1,model+'.pt') for model in model_list]
    graph_list2 = [os.path.join(g_path_2,model+'.pt') for model in model_list]
    graph_list3 = [os.path.join(g_path_3,model+'.pt') for model in model_list]
    dataset = ThreeGraphDataset(graph_list1,graph_list2,graph_list3)
    data_loader = CustomDataLoader(dataset, batch_size=BATCH_SIZE,shuffle=False,num_workers=4)
    
    return data_loader