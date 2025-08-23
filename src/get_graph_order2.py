from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
import pandas as pd
import numpy as np
import os
from joblib import Parallel, delayed
from .utils import ChainResidueAtomDescriptor,ChainResidueAtomDescriptor_mock
from .utils import get_edge_df,run_dssp_simple,sequence_three_letter_one_hot
from torch_geometric import data as DATA
import torch
import re 
from collections import Counter
from .md import calculate_triangle_properties





def get_index(model_name,contact_dir,pdb_dir):
    pdb_path=os.path.join(pdb_dir,model_name+'.pdb')
    contact_file=os.path.join(contact_dir,model_name+'.txt')
    contact_df=pd.read_csv(contact_file,sep=' ',header=None,names=['contact','area'])
    
    # Filter out contacts whose area is zero (indicating they are far apart)
    contact_df = contact_df[contact_df["area"] != 0]
    contact_df['atom1']=contact_df['contact'].str.split('\t').str[0]
    contact_df['atom2']=contact_df['contact'].str.split('\t').str[1]

    # Extract residue-level ID
    contact_df['ID1'] = contact_df['atom1'].str.split('A<').str[0]
    contact_df['ID2'] = contact_df['atom2'].str.split('A<').str[0]

    # Create contact IDs
    contact_df['contact_id']=contact_df.apply(lambda row: f"{min(row['ID1'], row['ID2'])}\t{max(row['ID1'], row['ID2'])}", axis=1)
    contact_df = contact_df.groupby('contact_id').agg({
    'ID1': 'first',  
    'ID2': 'first'   
    }).reset_index()


    ID_sum=list(contact_df['ID1'])+list(contact_df['ID2'])
    ID_uniq=list(set(ID_sum))
    

    vertice_df=run_dssp_simple(pdb_path)
    vertice_df=vertice_df.loc[vertice_df['ID'].isin(ID_uniq)].copy()
    vertice_df['vertice_index'] = range(len(vertice_df))
    return vertice_df


def get_res_triangle(row,id_to_index):
    res_list=[row['res1_1'],row['res1_2'],row['res2_1'],row['res2_2']]
    res_list=list(set(res_list))
    mapped_values = [id_to_index.get(id, None) for id in res_list]
    result = [1, 1, 1] if None in mapped_values else sorted(mapped_values)
    return result

def count_atom_num(row):
    atom_list=[row['atom1_1'],row['atom1_2'],row['atom2_1'],row['atom2_2']]
    atom_num_dict = Counter(atom_list)
    atom_com = [key for key, value in atom_num_dict.items() if value > 1][0]
    uniq_list=[key for key, value in atom_num_dict.items() if value == 1]
    atom_uniq1,atom_uniq2= uniq_list[0], uniq_list[1]
    return atom_com,atom_uniq1,atom_uniq2
    
def get_prop(row,atom_df):
    point_1 = atom_df.loc[atom_df['atom'] == row['atom_com'], ['co_1', 'co_2', 'co_3']].iloc[0].values
    point_2 = atom_df.loc[atom_df['atom'] == row['atom_uniq1'], ['co_1', 'co_2', 'co_3']].iloc[0].values
    point_3 = atom_df.loc[atom_df['atom'] == row['atom_uniq2'], ['co_1', 'co_2', 'co_3']].iloc[0].values
    pps = calculate_triangle_properties(point_1, point_2, point_3)
    return pps


def energy_res_contact(model_name,slice_dir,interface_dir,energy_df_2order):
    atom_file=os.path.join(interface_dir,model_name+'.txt')
    atom_df=pd.read_csv(atom_file,sep=' ',header=None,names=['atom','co_1','co_2','co_3'])
    IDs=list(atom_df['atom'])

    slice_file=os.path.join(slice_dir,model_name+'.txt')
    with open(slice_file, 'r') as file:
        lines = file.readlines()

    slice_2_data = []
    slice_2_found = False
    for line in lines:
        if 'Slice 3:' in line:
            break
        if 'Slice 2:' in line:
            slice_2_found = True
            continue
        if slice_2_found:
            line = line.strip()
            if line:
                slice_2_data.append(line)

    node_data = []
    edge_data = []

    for entry in slice_2_data:
        try:
            tuple_data = eval(entry)
            first_array_length = len(tuple_data[0])
            if first_array_length == 1:
                node_data.append(tuple_data[0])
            elif first_array_length == 2:
                edge_data.append(tuple_data)
            # Length 1 corresponding to nodes, length 2 corresponding to edges;
            # higher dimensions (triangles, etc.) are ignored
            elif first_array_length > 2:
                break
        except Exception as e:
            print(f"Error parsing line: {entry}, Error: {e}")
    
    edge_list=[]
    for edge in edge_data:
        e1=edge[0][0];e2=edge[0][1];birth=edge[1]
        e1_1=e1[0];e1_2=e1[1];e2_1=e2[0];e2_2=e2[1]

        ##skip edge with same chain
        if IDs[e1_1][2]==IDs[e1_2][2] or IDs[e2_1][2]==IDs[e2_2][2]:
            continue
        edge_list.append({
            'pair1':f"{IDs[e1_1]}\t{IDs[e1_2]}",
            'pair2':f"{IDs[e2_1]}\t{IDs[e2_2]}",
            'birth':birth
        })
    contact_df=pd.DataFrame(edge_list)


    contact_df['atom1_1']=contact_df['pair1'].str.split('\t').str[0]
    contact_df['atom1_2']=contact_df['pair1'].str.split('\t').str[1]
    contact_df['atom2_1']=contact_df['pair2'].str.split('\t').str[0]
    contact_df['atom2_2']=contact_df['pair2'].str.split('\t').str[1]

    contact_df['atom_type1_1']=contact_df['atom1_1'].apply(lambda x: ChainResidueAtomDescriptor(x).generalize_name())
    contact_df['atom_type1_2']=contact_df['atom1_2'].apply(lambda x: ChainResidueAtomDescriptor(x).generalize_name())
    contact_df['atom_type2_1']=contact_df['atom2_1'].apply(lambda x: ChainResidueAtomDescriptor(x).generalize_name())
    contact_df['atom_type2_2']=contact_df['atom2_2'].apply(lambda x: ChainResidueAtomDescriptor(x).generalize_name())
    contact_df['pair1']=contact_df.apply(lambda row: f"{min(row['atom_type1_1'],row['atom_type1_2'])}\t{max(row['atom_type1_1'],row['atom_type1_2'])}",axis=1)
    contact_df['pair2']=contact_df.apply(lambda row: f"{min(row['atom_type2_1'],row['atom_type2_2'])}\t{max(row['atom_type2_1'],row['atom_type2_2'])}",axis=1)
    contact_df['pair']=contact_df.apply(lambda row: f"{min(row['pair1'],row['pair2'])}\t{max(row['pair1'],row['pair2'])}",axis=1)
    contact_df = contact_df.merge(energy_df_2order[['pair', 'energy']], on='pair', how='left')
    
    contact_df['res1_1']=contact_df['atom1_1'].str.split('A<').str[0]
    contact_df['res1_2']=contact_df['atom1_2'].str.split('A<').str[0]
    contact_df['res2_1']=contact_df['atom2_1'].str.split('A<').str[0]
    contact_df['res2_2']=contact_df['atom2_2'].str.split('A<').str[0]
    contact_df['pair1_res']=contact_df.apply(lambda row: f"{min(row['res1_1'],row['res1_2'])}\t{max(row['res1_1'],row['res1_2'])}", axis=1)
    contact_df['pair2_res']=contact_df.apply(lambda row: f"{min(row['res2_1'],row['res2_2'])}\t{max(row['res2_1'],row['res2_2'])}", axis=1)
    contact_df['contact_id']=contact_df.apply(lambda row: f"{min(row['pair1_res'], row['pair2_res'])}\t{max(row['pair1_res'], row['pair2_res'])}", axis=1)
    
    # Aggregate from atom-level contact to residue-level contact
    result_df = contact_df.groupby('contact_id').agg({
    'energy':'mean',
    'birth':'mean',
    'pair1_res': 'first', 
    'pair2_res': 'first'  
    }).reset_index()
    result_df.rename(columns={'pair1_res_first': 'pair1_res','pair2_res_first': 'pair2_res'}, inplace=True)   
    col=result_df.columns.tolist()[1:-2]
    result_df.fillna(value=0, inplace=True) 

    return result_df,col


def energy_res_order1_contact(model_name,pdb_dir,contact_dir,energy_df,energy_df_mock,atom_lookup): 
    pdb_file=os.path.join(pdb_dir,model_name+'.pdb')

    contact_file=os.path.join(contact_dir,model_name+'.txt')
    contact_df=pd.read_csv(contact_file,sep=' ',header=None,names=['contact','area'])
    contact_df=contact_df[contact_df['area']!=0]
    contact_df['atom1']=contact_df['contact'].str.split('\t').str[0]
    contact_df['atom2']=contact_df['contact'].str.split('\t').str[1]

    contact_df['atom_type1'] = contact_df['atom1'].apply(lambda x: ChainResidueAtomDescriptor(x).generalize_name())
    contact_df['atom_type2'] = contact_df['atom2'].apply(lambda x: ChainResidueAtomDescriptor(x).generalize_name())
    contact_df['atom_type1_m'] = contact_df['atom1'].apply(lambda x: ChainResidueAtomDescriptor_mock(x,atom_lookup).generalize_name())
    contact_df['atom_type2_m'] = contact_df['atom2'].apply(lambda x: ChainResidueAtomDescriptor_mock(x,atom_lookup).generalize_name())
    contact_df['pair'] = contact_df.apply(lambda row: f"{min(row['atom_type1'], row['atom_type2'])}\t{max(row['atom_type1'], row['atom_type2'])}", axis=1)
    contact_df['pair_mock'] = contact_df.apply(lambda row: f"{min(row['atom_type1_m'], row['atom_type2_m'])}\t{max(row['atom_type1_m'], row['atom_type2_m'])}", axis=1)

    contact_df = contact_df.merge(energy_df[['pair', 'energy']], on='pair', how='left')
    contact_df = contact_df.merge(energy_df_mock[['pair_mock', 'energy_mock']], on='pair_mock', how='left')


    contact_df['ID1'] = contact_df['atom1'].str.split('A<').str[0]
    contact_df['ID2'] = contact_df['atom2'].str.split('A<').str[0]

    contact_df['area_energy'] = contact_df['energy'] * contact_df['area']
    contact_df['area_energy_mock'] = contact_df['energy_mock'] * contact_df['area']


    contact_df['contact_id_order1']=contact_df.apply(lambda row: f"{min(row['ID1'], row['ID2'])}\t{max(row['ID1'], row['ID2'])}", axis=1)

    result_df = contact_df.groupby('contact_id_order1').agg({
    'area': 'sum',
    'energy': ['sum'],   
    'area_energy': ['sum'],
    'energy_mock':['sum'],
    'area_energy_mock':['sum'],
    'ID1': 'first',  
    'ID2': 'first'   
    }).reset_index()
    result_df.columns = ['_'.join(col).rstrip('_') for col in result_df.columns]
    result_df.rename(columns={'ID1_first': 'ID1','ID2_first':'ID2'}, inplace=True)   
    col=result_df.columns.tolist()[1:-2]
    result_df.fillna(value=0, inplace=True) 


    res1=list(result_df['ID1'])
    res1=[re.search(r'R<([^>]+)>', res).group(1) for res in res1]
    one1=sequence_three_letter_one_hot(res1)
    res2=list(result_df['ID2'])
    res2=[re.search(r'R<([^>]+)>', res).group(1) for res in res2]
    one2=sequence_three_letter_one_hot(res2)
   

    one_hot_df = pd.DataFrame(one1+one2,
                            columns=[f'AA_{i}' for i in range(21)])
    
    edge_df=get_edge_df(result_df,pdb_file,normal=False)
    dis_col=[f'dis_{i}' for i in range(11)]
    result_df = pd.concat([result_df, one_hot_df,edge_df], axis=1)
    col=col+['AA_'+str(i) for i in range(21)]+dis_col

    
    return result_df,col



def get_vertice_and_edge(model_name,pdb_dir,slice_dir,contact_dir,interface_dir,energy_df,energy_df_mock,energy_df_2order,atom_lookup):
    order1_vertice=get_index(model_name,contact_dir,pdb_dir)
    contact_df,edge_col=energy_res_contact(model_name,slice_dir,interface_dir,energy_df_2order)
    contact_df_order1,vertice_col=energy_res_order1_contact(model_name,pdb_dir,contact_dir,energy_df,energy_df_mock,atom_lookup)

    
    ID_sum=list(contact_df['pair1_res'])+list(contact_df['pair2_res'])
    ID_uniq=sorted(list(set(ID_sum)))
    vertice_df=pd.DataFrame(ID_uniq,columns=['contact_id_order1'])


    # Obtain the list of nodes
    vertice_df=vertice_df.merge(contact_df_order1[['contact_id_order1']+vertice_col+['ID1','ID2']],on='contact_id_order1',how='inner')
    vertice_df = vertice_df.fillna(0)   
    vertice_df.loc[:,'vertice_index'] = range(len(vertice_df))
    order1_vertice.rename(columns={"ID": "ID1", "vertice_index": "vertice_index1"}, inplace=True)
    vertice_df = vertice_df.merge(order1_vertice, on="ID1", how="left")
    order1_vertice.rename(columns={"ID1": "ID2", "vertice_index1": "vertice_index2"}, inplace=True)
    vertice_df = vertice_df.merge(order1_vertice, on="ID2", how="left")    
    order1_vertice.rename(columns={"ID2": "ID", "vertice_index2": "vertice_index"}, inplace=True)



    # Obtain the indices of edges
    tmp_df = pd.merge(contact_df, vertice_df[['contact_id_order1','vertice_index']], left_on='pair1_res', right_on='contact_id_order1', how='inner')
    tmp_df.rename(columns={'vertice_index': 'vertice_index1'}, inplace=True)
    del tmp_df['contact_id_order1']
    merged_df = pd.merge(tmp_df, vertice_df[['contact_id_order1','vertice_index']], left_on='pair2_res', right_on='contact_id_order1', how='inner')
    merged_df.rename(columns={'vertice_index': 'vertice_index2'}, inplace=True)
    del merged_df['contact_id_order1']


    reverse_df = merged_df.rename(columns={'vertice_index1': 'vertice_index2', 'vertice_index2': 'vertice_index1'})
    full_df = pd.concat([merged_df, reverse_df], ignore_index=True)
    full_df = full_df.drop_duplicates()

    full_df['res1_1']=full_df['pair1_res'].str.split('\t').str[0]
    full_df['res1_2']=full_df['pair1_res'].str.split('\t').str[1]
    full_df['res2_1']=full_df['pair2_res'].str.split('\t').str[0]
    full_df['res2_2']=full_df['pair2_res'].str.split('\t').str[1]
    id_to_index = dict(zip(order1_vertice['ID'], order1_vertice['vertice_index']))
    full_df['triangle_index']=full_df.apply(lambda row: get_res_triangle(row,id_to_index), axis=1)
    full_df['triangle_index'] = full_df['triangle_index'].apply(lambda x: [1, 1, 1] if len(x) == 2 else x)
    
    return vertice_df,full_df,vertice_col,edge_col




def permodel_df(model_name,pdb_dir,contact_dir,slice_dir,interface_dir,graph_dir,energy_df,energy_df_mock,energy_df_2order,atom_lookup,dataname='test'):
    try:
        vertice_df,edge_df,vertice_col,edge_col=get_vertice_and_edge(model_name,pdb_dir,slice_dir,contact_dir,interface_dir,energy_df,energy_df_mock,energy_df_2order,atom_lookup)
        fea=vertice_df[vertice_col].values
        triangle_array = np.array(edge_df['triangle_index'].tolist(), dtype=np.int64)
        

        GCNData =DATA.Data(x=torch.tensor(fea,dtype=torch.float32),
                                            edge_index=torch.tensor(edge_df[['vertice_index1', 'vertice_index2']].values.T, dtype=torch.long),
                                            edge_attr=torch.tensor(edge_df[edge_col].values,dtype=torch.float32),
                                            node_index=torch.tensor(vertice_df[['vertice_index1', 'vertice_index2']].values.T, dtype=torch.long),
                                            triangle_index=torch.tensor(triangle_array.T,dtype=torch.long))
        GCNData.__setitem__('model_name', [dataname+'&'+model_name])

        graph_path = os.path.join(graph_dir,model_name+'.pt')
        torch.save(GCNData,graph_path)
    except Exception as e:
        print(f"error in {model_name}: {e}")

def batch_graph_order2(pdb_dir,contact_dir,slice_dir,interface_dir,graph_dir,energy_df,energy_df_mock,energy_df_2order,atom_lookup,n_jobs,dataname='test'):
    model_list=[file.split('.')[0] for file in os.listdir(contact_dir)]
    Parallel(n_jobs=n_jobs)(
    delayed(permodel_df)(model,pdb_dir,contact_dir,slice_dir,interface_dir,graph_dir,energy_df,energy_df_mock,energy_df_2order,atom_lookup,dataname) for model in model_list)





