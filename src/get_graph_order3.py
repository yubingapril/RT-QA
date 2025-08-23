import os
import pandas as pd 
from .utils import ChainResidueAtomDescriptor,run_dssp_simple
from sklearn.preprocessing import MinMaxScaler
from torch_geometric import data as DATA
import torch
from joblib import Parallel,delayed
import numpy as np
from .md import calculate_triangle_properties_1,calculate_dihedral_angle_from_planes
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP



def get_index_order1(model_name,contact_dir,pdb_dir):
    pdb_path=os.path.join(pdb_dir,model_name+'.pdb')
    contact_file=os.path.join(contact_dir,model_name+'.txt')
    contact_df=pd.read_csv(contact_file,sep=' ',header=None,names=['contact','area'])
    
    contact_df = contact_df[contact_df["area"] != 0]
    contact_df['atom1']=contact_df['contact'].str.split('\t').str[0]
    contact_df['atom2']=contact_df['contact'].str.split('\t').str[1]


    contact_df['ID1'] = contact_df['atom1'].str.split('A<').str[0]
    contact_df['ID2'] = contact_df['atom2'].str.split('A<').str[0]

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



def process_voro_file(model_name,voro_dir,atom_coor_dir,energy_df,graph_dir,dataname,contact_dir,pdb_dir):
    try:
        voro_file=os.path.join(voro_dir,model_name+'.txt')
        atom_coor_file=os.path.join(atom_coor_dir,model_name+'.txt')

        with open(voro_file, 'r') as file:
            lines = file.readlines()

        slice_3_data = []
        slice_3_found = False
        for line in lines:
            if 'Slice 3:' in line:
                slice_3_found = True
                continue
            if slice_3_found:
                line = line.strip()
                if line:
                    slice_3_data.append(line)

        one_dimensional_data = []
        two_dimensional_data = []

        for entry in slice_3_data:
            try:
                tuple_data = eval(entry)
                first_array_length = len(tuple_data[0])
                if first_array_length == 1:
                    one_dimensional_data.append(tuple_data)
                elif first_array_length == 2:
                    two_dimensional_data.append(tuple_data)
                # Length 1 corresponding to nodes, length 2 corresponding to edges;
                # higher dimensions (triangles, etc.) are ignored    
                elif first_array_length > 2:
                    break
            except Exception as e:
                print(f"Error parsing line: {entry}, Error: {e}")

        atom_coor_df=pd.read_csv(atom_coor_file,sep=' ',names=['ID','co_1','co_2','co_3'])


        node_data=one_dimensional_data;edge_data=two_dimensional_data
        node_data1=[]
        node_data_list=[]
        for node in node_data:
            # print(node)
            node_num=node[0][0]
            ID1,ID2,ID3=atom_coor_df['ID'][node_num[0]],atom_coor_df['ID'][node_num[1]],atom_coor_df['ID'][node_num[2]]

            res1=ID1.split('A<')[0];res2=ID2.split('A<')[0];res3=ID3.split('A<')[0]
            
            chain1, chain2, chain3= ID1[2], ID2[2],ID3[2]
            
            if chain1==chain2==chain3:
                continue
            if len({res1, res2, res3}) < 3:
                continue
            node_data1.append(node)
            
            res_sort = sorted([res1, res2, res3])
            node_type = f"{res_sort[0]}\t{res_sort[1]}\t{res_sort[2]}"

            point_1 = atom_coor_df.loc[node_num[0], ['co_1', 'co_2', 'co_3']].values
            point_2 = atom_coor_df.loc[node_num[1], ['co_1', 'co_2', 'co_3']].values
            point_3 = atom_coor_df.loc[node_num[2], ['co_1', 'co_2', 'co_3']].values
            pps = calculate_triangle_properties_1(point_1, point_2, point_3)
            
            node_data_list.append({
                'node_type': node_type,
                'res1':res1,
                'res2':res2,
                'res3':res3,
                'ID1': ID1,
                'ID2': ID2,
                'ID3': ID3,
                'prop1': pps[0],
                'prop2': pps[1],
                'prop3': pps[2],
                'prop4': pps[3],
                'prop5': pps[4],
                'prop6': pps[5],
                'prop7': pps[6],
                'prop8': pps[7],
                'prop9': pps[8],
                'prop10':pps[9]
            })
        node_df = pd.DataFrame(node_data_list)


        node_df['atom_type1'] = node_df['ID1'].apply(lambda x: ChainResidueAtomDescriptor(x).generalize_name())
        node_df['atom_type2'] = node_df['ID2'].apply(lambda x: ChainResidueAtomDescriptor(x).generalize_name())
        node_df['atom_type3'] = node_df['ID3'].apply(lambda x: ChainResidueAtomDescriptor(x).generalize_name())

        node_df['pair1'] = node_df.apply(lambda row: f"{min(row['atom_type1'], row['atom_type2'])}\t{max(row['atom_type1'], row['atom_type2'])}", axis=1)
        node_df['pair2'] = node_df.apply(lambda row: f"{min(row['atom_type2'], row['atom_type3'])}\t{max(row['atom_type2'], row['atom_type3'])}", axis=1)
        node_df['pair3'] = node_df.apply(lambda row: f"{min(row['atom_type1'], row['atom_type3'])}\t{max(row['atom_type1'], row['atom_type3'])}", axis=1)

        node_df = node_df.merge(energy_df[['pair', 'energy']], left_on='pair1', right_on='pair',how='left')
        node_df = node_df.rename(columns={'energy': 'energy1'})
        node_df = node_df.merge(energy_df[['pair', 'energy']], left_on='pair2', right_on='pair',how='left')
        node_df = node_df.rename(columns={'energy': 'energy2'})
        node_df = node_df.merge(energy_df[['pair', 'energy']], left_on='pair3', right_on='pair',how='left')
        node_df = node_df.rename(columns={'energy': 'energy3'})
        node_df['energy']=node_df['energy1']+node_df['energy2']+node_df['energy3']

        node_df_res = node_df.groupby('node_type').agg({
        'res1':'first',
        'res2':'first',
        'res3':'first',
        'energy': ['min','max','sum','mean'],
        'prop1':'mean',
        'prop2':'mean',
        'prop3':'mean',
        'prop4':'mean',
        'prop5':'mean',
        'prop6':'mean',
        'prop7':'mean',
        'prop8':'mean',
        'prop9':'mean',
        'prop10':'mean',
        'node_type': 'count',
        }).reset_index()
        

        prop_col=[f"prop{i+1}" for i in range(10)]
        node_df_res.columns = ['node_types', 'res1','res2','res3','energy_min', 'energy_max', 'energy_sum', 'energy_mean']+prop_col+['count']
        
        res_index_df=get_index_order1(model_name,contact_dir,pdb_dir)
        res_index_df.rename(columns={"ID": "res1", "vertice_index": "vertice_index1"}, inplace=True)
        node_df_res = node_df_res.merge(res_index_df, on="res1", how="left")
        res_index_df.rename(columns={"res1": "res2", "vertice_index1": "vertice_index2"}, inplace=True)
        node_df_res = node_df_res.merge(res_index_df, on="res2", how="left")            
        res_index_df.rename(columns={"res2": "res3", "vertice_index2": "vertice_index3"}, inplace=True)
        node_df_res = node_df_res.merge(res_index_df, on="res3", how="left")  

        node_df_res['triangle_index'] = node_df_res.apply(
            lambda row: list(map(int, sorted([row['vertice_index1'], row['vertice_index2'], row['vertice_index3']])))
            if row[['vertice_index1', 'vertice_index2', 'vertice_index3']].isna().sum() == 0 
            else [0, 0, 0],
            axis=1
        )

    
        node_col=['energy_min', 'energy_max', 'energy_sum', 'energy_mean','count']+prop_col

        scaler=MinMaxScaler()
        node_df_res[node_col]=scaler.fit_transform(node_df_res[node_col])
        

        node_df_res.fillna(0, inplace=True)

        ####edge 
        node_list=[item[0][0] for item in node_data1]
        edge_data_list=[]
        for edge in edge_data:
            edge1, edge2 = edge[0][0], edge[0][1]
            
            if edge1 not in node_list or edge2 not in node_list:
                continue
            
            ID1_1,ID1_2,ID1_3=atom_coor_df['ID'][edge1[0]],atom_coor_df['ID'][edge1[1]],atom_coor_df['ID'][edge1[2]]
            ID2_1,ID2_2,ID2_3=atom_coor_df['ID'][edge2[0]],atom_coor_df['ID'][edge2[1]],atom_coor_df['ID'][edge2[2]]
            res1_1=ID1_1.split('A<')[0];res1_2=ID1_2.split('A<')[0];res1_3=ID1_3.split('A<')[0]
            res2_1=ID2_1.split('A<')[0];res2_2=ID2_2.split('A<')[0];res2_3=ID2_3.split('A<')[0]  
            
            res_sort = sorted([res1_1, res1_2, res1_3])
            node_type1 = f"{res_sort[0]}\t{res_sort[1]}\t{res_sort[2]}"
            res_sort = sorted([res2_1, res2_2, res2_3])
            node_type2 = f"{res_sort[0]}\t{res_sort[1]}\t{res_sort[2]}"
            edge_type=f"{min(node_type1,node_type2)}\t\t{max(node_type1,node_type2)}"

            point1 = atom_coor_df.loc[edge1[0],['co_1', 'co_2', 'co_3']].values 
            point2 = atom_coor_df.loc[edge1[1],['co_1', 'co_2', 'co_3']].values 
            point3 = atom_coor_df.loc[edge1[2],['co_1', 'co_2', 'co_3']].values 
            point4 = atom_coor_df.loc[edge2[0],['co_1', 'co_2', 'co_3']].values 
            point5 = atom_coor_df.loc[edge2[1],['co_1', 'co_2', 'co_3']].values
            point6 = atom_coor_df.loc[edge2[2],['co_1', 'co_2', 'co_3']].values
            # ang=calculate_dihedral_angle_from_planes_abs(point1,point2,point3,point4,point5,point6)
            # print(ang)
            tuple1 = (ID1_1,ID1_2,ID1_3)
            tuple2 = (ID2_1,ID2_2,ID2_3)
            if tuple1<tuple2: # Determine input order of faces
                ang=calculate_dihedral_angle_from_planes(point1,point2,point3,point4,point5,point6)
            else:
                ang=calculate_dihedral_angle_from_planes(point4,point5,point6,point1,point2,point3)

            edge_data_list.append({
                'edge_type': edge_type,
                'ang':ang
            })        


        edge_df = pd.DataFrame(edge_data_list)
        edge_df_res = edge_df.groupby('edge_type').agg({
        'edge_type': 'count',
        'ang':['mean']
        }).reset_index()
        edge_df_res.columns = ['edge_type', 'count','ang']

        edge_df_res['node_type1']=edge_df_res['edge_type'].str.split('\t\t').str[0]
        edge_df_res['node_type2']=edge_df_res['edge_type'].str.split('\t\t').str[1]
        
        edge_df_res = edge_df_res.merge(node_df_res.reset_index()[['node_types','index']], left_on='node_type1',right_on='node_types', how='left')
        edge_df_res.rename(columns={'index': 'index1'}, inplace=True)
        edge_df_res.drop(columns=['node_types'], inplace=True)
        edge_df_res = edge_df_res.merge(node_df_res.reset_index()[['node_types','index']], left_on='node_type2',right_on='node_types', how='left')
        edge_df_res.rename(columns={'index': 'index2'}, inplace=True)
        edge_df_res.drop(columns=['node_types'], inplace=True)

        edge_col= ['count','ang']
        edge_df_res[edge_col]=scaler.fit_transform(edge_df_res[edge_col])


        reverse_edge_df = edge_df_res.rename(columns={'index1': 'index2', 'index2': 'index1'})
        edge_df_res=pd.concat([edge_df_res,reverse_edge_df],ignore_index=True)

        edge_index=torch.tensor(edge_df_res[['index1', 'index2']].values.T, dtype=torch.long)
        edge_attr=torch.tensor(edge_df_res[edge_col].values, dtype=torch.float32)
        fea=node_df_res[node_col].values
        triangle_array = np.array(node_df_res['triangle_index'].tolist(), dtype=np.int64)

        GCNData =DATA.Data(x=torch.tensor(fea,dtype=torch.float32),
                                            edge_index=edge_index,
                                            edge_attr=edge_attr,
                                            triangle_index=torch.tensor(triangle_array.T, dtype=torch.long)
                                            )
        GCNData.__setitem__('model_name', [dataname+'&'+model_name])
        graph_path = os.path.join(graph_dir,model_name+'.pt')
        torch.save(GCNData,graph_path)
    except Exception as e:
        print(f"error in {model_name}: {e}")
    return None



def batch_graph_order3(slice_dir,interface_dir,energy_df,graph_dir,contact_dir,pdb_dir,n_jobs,dataname='test'):
    model_list=[file.split('.')[0] for file in os.listdir(contact_dir)]
    Parallel(n_jobs=n_jobs)(
        delayed(process_voro_file)(model,slice_dir,interface_dir,energy_df,graph_dir,dataname,contact_dir,pdb_dir) for model in model_list)
    