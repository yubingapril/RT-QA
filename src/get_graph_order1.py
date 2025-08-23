from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
from joblib import Parallel, delayed
from .utils import ChainResidueAtomDescriptor,ChainResidueAtomDescriptor_mock
from .utils import get_edge_fea_energy_col_nonormal,get_all_col,sequence_three_letter_one_hot,ss8_one_hot,run_dssp
from torch_geometric import data as DATA
import torch




def energy_res_contact(model_name,contact_dir,energy_df,energy_df_mock,atom_lookup):


    contact_file=os.path.join(contact_dir,model_name+'.txt')
    contact_df=pd.read_csv(contact_file,sep=' ',header=None,names=['contact','area'])
    # Remove contacts with zero area
    contact_df = contact_df[contact_df["area"] != 0]
    contact_df['atom1']=contact_df['contact'].str.split('\t').str[0]
    contact_df['atom2']=contact_df['contact'].str.split('\t').str[1]

    contact_df['atom_type1'] = contact_df['atom1'].apply(lambda x: ChainResidueAtomDescriptor(x).generalize_name())
    contact_df['atom_type2'] = contact_df['atom2'].apply(lambda x: ChainResidueAtomDescriptor(x).generalize_name())
    contact_df['atom_type1_mock'] = contact_df['atom1'].apply(lambda x: ChainResidueAtomDescriptor_mock(x,atom_lookup).generalize_name())
    contact_df['atom_type2_mock'] = contact_df['atom2'].apply(lambda x: ChainResidueAtomDescriptor_mock(x,atom_lookup).generalize_name())
    
    # Create a column for pairs
    contact_df['pair'] = contact_df.apply(lambda row: f"{min(row['atom_type1'], row['atom_type2'])}\t{max(row['atom_type1'], row['atom_type2'])}", axis=1)
    contact_df['pair_mock'] = contact_df.apply(lambda row: f"{min(row['atom_type1_mock'], row['atom_type2_mock'])}\t{max(row['atom_type1_mock'], row['atom_type2_mock'])}", axis=1)

    # Merge contact_df and energy_df to obtain energy values
    contact_df = contact_df.merge(energy_df[['pair', 'energy']], on='pair', how='left')
    contact_df = contact_df.merge(energy_df_mock[['pair_mock', 'energy_mock']], on='pair_mock', how='left')   

    contact_df['ID1'] = contact_df['atom1'].str.split('A<').str[0]
    contact_df['ID2'] = contact_df['atom2'].str.split('A<').str[0]

    # Calculate the product of area and energy
    contact_df['area_energy'] = contact_df['energy'] * contact_df['area']
    contact_df['area_energy_mock'] = contact_df['energy_mock'] * contact_df['area']

    # Create a flag column for contacts
    contact_df['contact_id']=contact_df.apply(lambda row: f"{min(row['ID1'], row['ID2'])}\t{max(row['ID1'], row['ID2'])}", axis=1)

    # Aggregate from atom-level contact to residue-level contact
    result_df = contact_df.groupby('contact_id').agg({
    'area': 'sum',
    'energy': ['sum'],   
    'area_energy': ['sum'],
    'energy_mock': ['sum'], 
    'area_energy_mock': ['sum'],
    'ID1': 'first',  
    'ID2': 'first'   
    }).reset_index()
    result_df.columns = ['_'.join(col).rstrip('_') for col in result_df.columns]
    result_df.rename(columns={'ID1_first': 'ID1','ID2_first':'ID2'}, inplace=True)   
    col=result_df.columns.tolist()[1:-2]
    result_df.fillna(value=0, inplace=True)  
    

    return result_df,col



def get_vertice_and_edge(model_name,contact_dir,pdb_dir,energy_df,energy_df_mock,atom_lookup):
    pdb_path=os.path.join(pdb_dir,model_name+'.pdb')
    contact_df,col=energy_res_contact(model_name,contact_dir,energy_df,energy_df_mock,atom_lookup)

    graph_vertice_df=run_dssp(pdb_path)
    
    ID_sum=list(contact_df['ID1'])+list(contact_df['ID2'])
    ID_uniq=list(set(ID_sum))

    graph_vertice_df_filter=graph_vertice_df.loc[graph_vertice_df['ID'].isin(ID_uniq)].copy()



    graph_vertice_df_filter = graph_vertice_df_filter.copy()
    graph_vertice_df_filter['vertice_index'] = range(len(graph_vertice_df_filter))
    
    # Get edge indices
    tmp_df = pd.merge(contact_df, graph_vertice_df_filter[['ID','vertice_index']], left_on='ID1', right_on='ID', how='inner')
    tmp_df.rename(columns={'vertice_index': 'vertice_index1'}, inplace=True)
    del tmp_df['ID']
    merged_df = pd.merge(tmp_df, graph_vertice_df_filter[['ID','vertice_index']], left_on='ID2', right_on='ID', how='inner')
    merged_df.rename(columns={'vertice_index': 'vertice_index2'}, inplace=True)
    del merged_df['ID']
    
    # Compute initial edge features
    edge_index,edge_attr=get_edge_fea_energy_col_nonormal(merged_df,pdb_path,col)
    return graph_vertice_df_filter,edge_index,edge_attr




def permodel_df(model_name,pdb_dir,contact_dir,graph_dir,dataname,energy_df,energy_df_mock,atom_lookup):
    try:
        graph_vertice_df_filter, edge_index,edge_attr=get_vertice_and_edge(model_name,contact_dir,pdb_dir,energy_df,energy_df_mock,atom_lookup)
        fea_col=get_all_col(topo=False)

        fea=graph_vertice_df_filter[fea_col].values


        GCNData =DATA.Data(x=torch.tensor(fea,dtype=torch.float32),
                                            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                            edge_attr=torch.tensor(edge_attr,dtype=torch.float32))
        GCNData.__setitem__('model_name', [dataname+'&'+model_name])

        graph_path = os.path.join(graph_dir,model_name+'.pt')
        torch.save(GCNData,graph_path)
    except Exception as e:
        print(f"error in {model_name}: {e}")

def batch_graph_order1(pdb_dir,contact_dir,graph_dir,energy_df,energy_df_mock,atom_lookup,n_jobs,dataname='test'):
    model_list=[file.split('.')[0] for file in os.listdir(contact_dir)]
    Parallel(n_jobs=n_jobs)(
    delayed(permodel_df)(model,pdb_dir,contact_dir,graph_dir,dataname,energy_df,energy_df_mock,atom_lookup) for model in model_list)

