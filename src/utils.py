import numpy as np
from sklearn.preprocessing import MinMaxScaler
import re
from Bio import PDB
import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP

def get_topo_col():
    e_set=[['C'], ['N'], ['O'], ['C', 'N'], ['C', 'O'], ['N', 'O'], ['C', 'N', 'O']]
    e_set_str=[''.join(element) if isinstance(element, list) else element for element in e_set]
    fea_col0=[f'{obj}_{stat}' for obj in ['death'] for stat in ['sum','min','max','mean','std']]
    col_0=[f'f0_{element}_{fea}' for element in e_set_str for fea in fea_col0]
    fea_col1=[f'{obj}_{stat}' for obj in ['len','birth','death'] for stat in ['sum','min','max','mean','std']]
    col_1=[f'f1_{element}_{fea}' for element in e_set_str for fea in fea_col1]
    topo_col=col_0+col_1
    return topo_col

def get_all_col(topo):
    basic_col=['SS8_0', 'SS8_1', 'SS8_2', 'SS8_3', 'SS8_4', 'SS8_5', 'SS8_6', 'SS8_7', 'AA_0', 'AA_1', 'AA_2', 'AA_3', 'AA_4', 'AA_5', 'AA_6', 'AA_7', 'AA_8', 'AA_9', 'AA_10', 'AA_11', 'AA_12', 'AA_13', 'AA_14', 'AA_15', 'AA_16', 'AA_17', 'AA_18', 'AA_19', 'AA_20','rasa', 'phi', 'psi']

    if topo:
       topo_col=get_topo_col()
       col = basic_col + topo_col
    else:
        col=basic_col
    return col


class inter_chain_dis(object):
    def Calculate_distance(Coor_df,arr_cutoff):
        Num_atoms = len(Coor_df)
        Distance_matrix_real = np.zeros((Num_atoms,Num_atoms),dtype=float) 
        Distance_matrix = np.ones((Num_atoms,Num_atoms),dtype=float)
        chain_list=list(Coor_df['ID'].str[2])
        for i in range(Num_atoms):
            for j in range(i,Num_atoms):
                if chain_list[i] == chain_list[j]:
                    Distance_matrix[i][j] = 0.0
                    Distance_matrix[j][i] = 0.0
                    continue
                x_i = float(Coor_df['co_1'][i])
                y_i = float(Coor_df['co_2'][i])
                z_i = float(Coor_df['co_3'][i])

                
                x_j = float(Coor_df['co_1'][j])
                y_j = float(Coor_df['co_2'][j])
                z_j = float(Coor_df['co_3'][j])  
                dis = np.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2 + (z_i - z_j) ** 2)
                if dis <= float(arr_cutoff[0]) or dis >= float(arr_cutoff[1]):
                    Distance_matrix[i][j] = 0.0
                    Distance_matrix[j][i] = 0.0
                else:
                    Distance_matrix[i][j] = 1.0
                    Distance_matrix[j][i] = 1.0
                    Distance_matrix_real[i][j] = dis
                    Distance_matrix_real[j][i] = dis

        return Distance_matrix,Distance_matrix_real
    



def get_pointcloud_type(descriptor1,descriptor2,model,e1,e2):
    ###e1 and e2 can be C N O or all etc.
    # Use regular expressions
    c_pattern = r'c<([^>]+)>'
    r_pattern = r'r<([^>]+)>'
    i_pattern = r'i<([^>]+)>'  # i< > is optional

    # Find matches
    c_match1 = re.search(c_pattern, descriptor1)
    r_match1 = re.search(r_pattern, descriptor1)
    i_match1 = re.search(i_pattern, descriptor1)
    c_match2 = re.search(c_pattern, descriptor2)
    r_match2 = re.search(r_pattern, descriptor2)
    i_match2 = re.search(i_pattern, descriptor2)

    # Extract the matched content; if the result is None, set the content to None
    c_content1=c_match1.group(1) if c_match1 else None 
    r_content1=int(r_match1.group(1)) if r_match1 else None
    i_content1=i_match1.group(1) if i_match1 else ' ' 
    c_content2=c_match2.group(1) if c_match2 else None 
    r_content2=int(r_match2.group(1)) if r_match2 else None
    i_content2=i_match2.group(1) if i_match2 else ' ' 

    res_id1=(' ',r_content1,i_content1)
    res1=model[c_content1][res_id1]
    res_id2=(' ',r_content2,i_content2)
    res2=model[c_content2][res_id2]

    # atom coord
    if e1=='all':
        atom_coords1 = [[float(atom.get_coord()[0]),float(atom.get_coord()[1]),
                                   float(atom.get_coord()[1])] for atom in res1.get_atoms()]
    else:
        atom_coords1 = [[float(atom.get_coord()[0]),float(atom.get_coord()[1]),
                                   float(atom.get_coord()[1])] for atom in res1.get_atoms() if atom.get_name()[0]==e1]
    atom_coords1 = np.array(atom_coords1)
    if e2=='all':
        atom_coords2 = [[float(atom.get_coord()[0]),float(atom.get_coord()[1]),
                                   float(atom.get_coord()[1])] for atom in res2.get_atoms()]
    else:    
        atom_coords2 = [[float(atom.get_coord()[0]),float(atom.get_coord()[1]),
                                   float(atom.get_coord()[1])] for atom in res2.get_atoms() if atom.get_name()[0]==e2]
    atom_coords2 = np.array(atom_coords2)
    return atom_coords1,atom_coords2

def distance_of_two_points(p1,p2):
    return np.linalg.norm(np.array(p1)-np.array(p2))


def get_dis_histogram(descriptor1,descriptor2,model,e1='all',e2='all'):
    point_cloud1,point_cloud2 = get_pointcloud_type(descriptor1,descriptor2,model,e1,e2)
    number_1=len(point_cloud1);number_2=len(point_cloud2)

    dis_list = sorted([distance_of_two_points(point_cloud1[ind_1], point_cloud2[ind_2]) for ind_1 in range(number_1) for ind_2 in range(number_2)])
    dis_list = np.array(dis_list)

    ## Define interval boundaries
    bins = np.arange(1,11,1)
    bins=np.append(bins,np.inf) # Add an infinite interval to include values greater than 10


    # Count the number of distances in each interval
    hist,_=np.histogram(dis_list,bins=bins)
    return hist



def get_atom_dis(vertice_df,model,edge):
    hist=get_dis_histogram(vertice_df['ID'][edge[0]],vertice_df['ID'][edge[1]],model)

    return hist.tolist()

def get_element_index_dis_atom(mat_re,mat,num,vertice_df_filter,model):
    arr_index = []
    edge_atrr=[]
    
    for i in range(len(mat)):
        for j in range(i+1,len(mat[i])):
            if float(mat[i][j]) == num:
                # All-atom distances
                hists=get_atom_dis(vertice_df_filter,model,[i,j])
                edge_atrr.append([mat_re[i][j]]+hists)
                edge_atrr.append([mat_re[i][j]]+hists)

                arr_index.append([i,j])
                arr_index.append([j,i])
    
    # Convert edge_attr to a NumPy array
    edge_atrr = np.array(edge_atrr)
    # Normalize edge attributes
    scaler = MinMaxScaler()
    edge_atrr = scaler.fit_transform(edge_atrr)
    return arr_index,edge_atrr


def cal_dis(descriptor1,descriptor2,model):
    c_pattern = r'c<([^>]+)>'
    r_pattern = r'r<([^>]+)>'
    i_pattern = r'i<([^>]+)>'  # i< > 是可选项

    #查找匹配
    c_match1 = re.search(c_pattern, descriptor1)
    r_match1 = re.search(r_pattern, descriptor1)
    i_match1 = re.search(i_pattern, descriptor1)
    c_match2 = re.search(c_pattern, descriptor2)
    r_match2 = re.search(r_pattern, descriptor2)
    i_match2 = re.search(i_pattern, descriptor2)

    #提取匹配的内容，如果匹配结果为None，则设置内容为None
    c_content1=c_match1.group(1) if c_match1 else None 
    r_content1=int(r_match1.group(1)) if r_match1 else None
    i_content1=i_match1.group(1) if i_match1 else ' ' 
    c_content2=c_match2.group(1) if c_match2 else None 
    r_content2=int(r_match2.group(1)) if r_match2 else None
    i_content2=i_match2.group(1) if i_match2 else ' ' 

    res_id1=(' ',r_content1,i_content1)
    res1=model[c_content1][res_id1]
    res_id2=(' ',r_content2,i_content2)
    res2=model[c_content2][res_id2]

    ca1 = res1['CA'] if 'CA' in res1 else list(res1.get_atoms())[0]
    ca2 = res2['CA'] if 'CA' in res2 else list(res2.get_atoms())[0]
    coord1 = ca1.coord
    coord2 = ca2.coord
    distance = np.linalg.norm(coord1 - coord2)

    
    ##cal_bins
    atom_coords1 = [[float(atom.get_coord()[0]),float(atom.get_coord()[1]),
                                   float(atom.get_coord()[1])] for atom in res1.get_atoms()]
    atom_coords2 = [[float(atom.get_coord()[0]),float(atom.get_coord()[1]),
                                   float(atom.get_coord()[1])] for atom in res2.get_atoms()]
    number_1=len(atom_coords1);number_2=len(atom_coords2)
    dis_list = sorted([distance_of_two_points(atom_coords1[ind_1], atom_coords2[ind_2]) for ind_1 in range(number_1) for ind_2 in range(number_2)])
    dis_list = np.array(dis_list)
    ##定义区间边界
    bins = np.arange(1,11,1)
    bins=np.append(bins,np.inf) ##添加一个无穷大区间用于包含大于10的值
    ##统计各区间的数量
    hist,_=np.histogram(dis_list,bins=bins)
    # print(len(hist))
    
    return [distance]+hist.tolist()



def get_edge_fea(edge_df,pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein",pdb_file)
    model=structure[0]

    arr_index = []
    edge_attr=[]
    
    for _,row in edge_df.iterrows():
        attr=cal_dis(row['ID1'],row['ID2'],model)
        # print(len(attr))
        index1=row['vertice_index1'];index2=row['vertice_index2']
        edge_attr.append(attr);edge_attr.append(attr)
        arr_index.append([index1,index2]);arr_index.append([index2,index1])
    
    edge_attr = np.array(edge_attr)
    # 标准化 edge_atrr
    scaler = MinMaxScaler()
    edge_attr = scaler.fit_transform(edge_attr)
    return arr_index,edge_attr

def get_edge_df(edge_df,pdb_file,normal=True):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein",pdb_file)
    model=structure[0]


    edge_attr=[]
    
    for _,row in edge_df.iterrows():
        attr=cal_dis(row['ID1'],row['ID2'],model)
        edge_attr.append(attr)

    edge_attr = np.array(edge_attr)
    # 标准化 edge_atrr
    if normal:
        scaler = MinMaxScaler()
        edge_attr = scaler.fit_transform(edge_attr)
    edge_attr_df=pd.DataFrame(edge_attr,columns=[f'dis_{i}' for i in range(11)])
    return edge_attr_df


def get_edge_fea_energy(edge_df,pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein",pdb_file)
    model=structure[0]

    arr_index = []
    edge_attr=[]
    
    for _,row in edge_df.iterrows():
        attr=cal_dis(row['ID1'],row['ID2'],model)
        attr=attr+row[['area', 'energy', 'area_energy']].tolist()
        # print(len(attr))
        index1=row['vertice_index1'];index2=row['vertice_index2']
        edge_attr.append(attr);edge_attr.append(attr)
        arr_index.append([index1,index2]);arr_index.append([index2,index1])
    
    edge_attr = np.array(edge_attr)
    # 标准化 edge_atrr
    scaler = MinMaxScaler()
    edge_attr = scaler.fit_transform(edge_attr)
    return arr_index,edge_attr

def get_edge_fea_energy_mock(edge_df,pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein",pdb_file)
    model=structure[0]

    arr_index = []
    edge_attr=[]
    
    for _,row in edge_df.iterrows():
        attr=cal_dis(row['ID1'],row['ID2'],model)
        attr=attr+row[['area', 'energy', 'area_energy','energy_mock','area_energy_mock']].tolist()
        # print(len(attr))
        index1=row['vertice_index1'];index2=row['vertice_index2']
        edge_attr.append(attr);edge_attr.append(attr)
        arr_index.append([index1,index2]);arr_index.append([index2,index1])
    
    edge_attr = np.array(edge_attr)
    # 标准化 edge_atrr
    scaler = MinMaxScaler()
    edge_attr = scaler.fit_transform(edge_attr)
    return arr_index,edge_attr


def get_edge_fea_energy_solvent(edge_df,pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein",pdb_file)
    model=structure[0]

    arr_index = []
    edge_attr=[]
    
    for _,row in edge_df.iterrows():
        attr=cal_dis(row['ID1'],row['ID2'],model)
        attr=attr+row[['area','energy','area_energy','solvent_energy1','area_solvent_energy1','solvent_energy2','area_solvent_energy2']].tolist()
        # print(len(attr))
        index1=row['vertice_index1'];index2=row['vertice_index2']
        edge_attr.append(attr);edge_attr.append(attr)
        arr_index.append([index1,index2]);arr_index.append([index2,index1])
    
    edge_attr = np.array(edge_attr)
    # 标准化 edge_atrr
    scaler = MinMaxScaler()
    edge_attr = scaler.fit_transform(edge_attr)
    return arr_index,edge_attr


def get_edge_fea_energy_solvent_mock(edge_df,pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein",pdb_file)
    model=structure[0]

    arr_index = []
    edge_attr=[]
    
    for _,row in edge_df.iterrows():
        attr=cal_dis(row['ID1'],row['ID2'],model)
        attr=attr+row[['area','energy','area_energy','solvent_energy1','area_solvent_energy1','solvent_energy2','area_solvent_energy2',
                       'energy_mock','area_energy_mock','solvent_energy1_mock','area_solvent_energy1_mock','solvent_energy2_mock','area_solvent_energy2_mock']].tolist()
        # print(len(attr))
        index1=row['vertice_index1'];index2=row['vertice_index2']
        edge_attr.append(attr);edge_attr.append(attr)
        arr_index.append([index1,index2]);arr_index.append([index2,index1])
    
    edge_attr = np.array(edge_attr)
    # 标准化 edge_atrr
    scaler = MinMaxScaler()
    edge_attr = scaler.fit_transform(edge_attr)
    return arr_index,edge_attr


def get_edge_fea_energy_col(edge_df,pdb_file,col):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein",pdb_file)
    model=structure[0]

    arr_index = []
    edge_attr=[]
    for _,row in edge_df.iterrows():
        attr=cal_dis(row['ID1'],row['ID2'],model)
        attr=attr+row[col].tolist()
        # print(len(attr))
        index1=row['vertice_index1'];index2=row['vertice_index2']
        edge_attr.append(attr);edge_attr.append(attr)
        arr_index.append([index1,index2]);arr_index.append([index2,index1])

    edge_attr = np.array(edge_attr)
    # 标准化 edge_atrr
    scaler = MinMaxScaler()
    edge_attr = scaler.fit_transform(edge_attr)
    return arr_index,edge_attr

def get_edge_fea_energy_col_nonormal(edge_df,pdb_file,col):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein",pdb_file)
    model=structure[0]

    arr_index = []
    edge_attr=[]
    for _,row in edge_df.iterrows():
        attr=cal_dis(row['ID1'],row['ID2'],model)
        attr=attr+row[col].tolist()
        # print(len(attr))
        index1=row['vertice_index1'];index2=row['vertice_index2']
        edge_attr.append(attr);edge_attr.append(attr)
        arr_index.append([index1,index2]);arr_index.append([index2,index1])

    edge_attr = np.array(edge_attr)
    return arr_index,edge_attr

def get_edge_fea_energy_col_cad(edge_df,pdb_file,col):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein",pdb_file)
    model=structure[0]

    arr_index = []
    edge_attr=[]
    cad=[]
    for _,row in edge_df.iterrows():
        attr=cal_dis(row['ID1'],row['ID2'],model)
        attr=attr+row[col].tolist()
        # print(len(attr))
        index1=row['vertice_index1'];index2=row['vertice_index2']
        edge_attr.append(attr);edge_attr.append(attr)
        arr_index.append([index1,index2]);arr_index.append([index2,index1])
        cad.append(row['cad']);cad.append(row['cad'])
    edge_attr = np.array(edge_attr)
    cad=np.array(cad).reshape(-1, 1)
    scaler = MinMaxScaler()
    edge_attr = scaler.fit_transform(edge_attr)
    return arr_index,edge_attr,cad


def get_edge_fea_energy_col_cad_no(edge_df,pdb_file,col):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein",pdb_file)
    model=structure[0]

    arr_index = []
    edge_attr=[]
    cad=[]
    for _,row in edge_df.iterrows():
        attr=cal_dis(row['ID1'],row['ID2'],model)
        attr=attr+row[col].tolist()
        index1=row['vertice_index1'];index2=row['vertice_index2']
        edge_attr.append(attr);edge_attr.append(attr)
        arr_index.append([index1,index2]);arr_index.append([index2,index1])
        cad.append(row['cad']);cad.append(row['cad'])
    edge_attr = np.array(edge_attr)
    cad=np.array(cad).reshape(-1, 1)
    return arr_index,edge_attr,cad

def get_edge_fea_energy_col_cad_nonoramal(edge_df,pdb_file,col):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein",pdb_file)
    model=structure[0]

    arr_index = []
    edge_attr=[]
    cad=[]
    for _,row in edge_df.iterrows():
        attr=cal_dis(row['ID1'],row['ID2'],model)
        attr=attr+row[col].tolist()
        index1=row['vertice_index1'];index2=row['vertice_index2']
        edge_attr.append(attr);edge_attr.append(attr)
        arr_index.append([index1,index2]);arr_index.append([index2,index1])
        cad.append(row['cad']);cad.append(row['cad'])
    edge_attr = np.array(edge_attr)
    cad=np.array(cad).reshape(-1, 1)
    return arr_index,edge_attr,cad


class ChainResidueAtomDescriptor:
    def __init__(self,crad):
        self.crad=crad

    def without_numbering(self):
        return self
    
    def extract_residue_and_atom(self):
        # Define regular expression pattern
        pattern = r'c<[^>]+>r<[^>]+>(?:i<[^>]+>)?R<([^>]+)>A<([^>]+)>'
        
        # Use regular expressions to find matches
        match = re.search(pattern, self.crad)
        
        if match:
            # Extract residue names and atom names
            residue_name = match.group(1)
            atom_name = match.group(2)
            return residue_name, atom_name
        
        return None, None

    def generalize_name(self):
        resName,name=self.extract_residue_and_atom()

        if name == "OXT":
            name = "O"

        if name in ["H1", "H2", "H3"]:
            name = "H"

        if resName == "MSE":
            resName = "MET"
            if name == "SE":
                name = "SD"

        if resName == "SEC":
            resName = "CYS"
            if name == "SE":
                name = "SG"

        if resName == "ALA" and name in ["HB1", "HB2", "HB3"]:
            name = "HB1"

        if resName == "ARG":
            if name in ["NH1", "NH2"]:
                name = "NH1"
            elif name in ["HB2", "HB3"]:
                name = "HB2"
            elif name in ["HG2", "HG3"]:
                name = "HG2"
            elif name in ["HD2", "HD3"]:
                name = "HD2"
            elif name in ["HH11", "HH12", "HH21", "HH22"]:
                name = "HH11"

        if resName == "ASP":
            if name in ["OD1", "OD2"]:
                name = "OD1"
            elif name in ["HB2", "HB3"]:
                name = "HB2"

        if resName == "ASN":
            if name in ["HB2", "HB3"]:
                name = "HB2"
            elif name in ["HD21", "HD22"]:
                name = "HD21"

        if resName == "CYS" and name in ["HB2", "HB3"]:
            name = "HB2"

        if resName == "GLU":
            if name in ["OE1", "OE2"]:
                name = "OE1"
            elif name in ["HB2", "HB3"]:
                name = "HB2"
            elif name in ["HG2", "HG3"]:
                name = "HG2"

        if resName == "GLN":
            if name in ["HB2", "HB3"]:
                name = "HB2"
            elif name in ["HG2", "HG3"]:
                name = "HG2"
            elif name in ["HE21", "HE22"]:
                name = "HE21"

        if resName == "GLY" and name in ["HA2", "HA3"]:
            name = "HA2"

        if resName == "HIS" and name in ["HB2", "HB3"]:
            name = "HB2"

        if resName == "ILE":
            if name in ["HG12", "HG13"]:
                name = "HG12"
            elif name in ["HG21", "HG22", "HG23"]:
                name = "HG21"
            elif name in ["HD11", "HD12", "HD13"]:
                name = "HD11"

        if resName == "LEU":
            if name in ["HB2", "HB3"]:
                name = "HB2"
            elif name in ["HD11", "HD12", "HD13"]:
                name = "HD11"
            elif name in ["HD21", "HD22", "HD23"]:
                name = "HD21"

        if resName == "LYS":
            if name in ["HB2", "HB3"]:
                name = "HB2"
            elif name in ["HG2", "HG3"]:
                name = "HG2"
            elif name in ["HD2", "HD3"]:
                name = "HD2"
            elif name in ["HE2", "HE3"]:
                name = "HE2"
            elif name in ["HZ1", "HZ2", "HZ3"]:
                name = "HZ1"

        if resName == "MET":
            if name in ["HB2", "HB3"]:
                name = "HB2"
            elif name in ["HG2", "HG3"]:
                name = "HG2"
            elif name in ["HE1", "HE2", "HE3"]:
                name = "HE1"

        if resName == "PHE":
            if name in ["CD1", "CD2"]:
                name = "CD1"
            elif name in ["CE1", "CE2"]:
                name = "CE1"
            elif name in ["HB2", "HB3"]:
                name = "HB2"
            elif name in ["HD1", "HD2"]:
                name = "HD1"
            elif name in ["HE1", "HE2"]:
                name = "HE1"

        if resName == "PRO":
            if name in ["HB2", "HB3"]:
                name = "HB2"
            elif name in ["HG2", "HG3"]:
                name = "HG2"
            elif name in ["HD2", "HD3"]:
                name = "HD2"

        if resName == "SER" and name in ["HB2", "HB3"]:
            name = "HB2"

        if resName == "THR" and name in ["HG21", "HG22", "HG23"]:
            name = "HG21"

        if resName == "TRP" and name in ["HB2", "HB3"]:
            name = "HB2"

        if resName == "TYR":
            if name in ["CD1", "CD2"]:
                name = "CD1"
            elif name in ["CE1", "CE2"]:
                name = "CE1"
            elif name in ["HB2", "HB3"]:
                name = "HB2"
            elif name in ["HD1", "HD2"]:
                name = "HD1"
            elif name in ["HE1", "HE2"]:
                name = "HE1"

        if resName == "VAL":
            if name in ["HG11", "HG12", "HG13"]:
                name = "HG11"
            elif name in ["HG21", "HG22", "HG23"]:
                name = "HG21"

        # return resName, name
        return f'R<{resName}>A<{name}>'




class ChainResidueAtomDescriptor_mock:
    def __init__(self,crad,atom_lookup):
        self.crad=crad
        self.atom_lookup=atom_lookup

    def without_numbering(self):
        return self
    
    def extract_residue_and_atom(self):
        pattern = r'c<[^>]+>r<[^>]+>(?:i<[^>]+>)?R<([^>]+)>A<([^>]+)>'
        
        match = re.search(pattern, self.crad)
        
        if match:
            residue_name = match.group(1)
            atom_name = match.group(2)
            return residue_name, atom_name
        
        return None, None

    def generalize_name(self):
        resName,name=self.extract_residue_and_atom()
        if name == "OXT":
            name = "O"

       
        if resName == "MSE":
            resName = "MET"
            if name == "SE":
                name = "SD"

        if resName == "SEC":
            resName = "CYS"
            if name == "SE":
                name = "SG"

        if resName == "ALA" and name in ["HB1", "HB2", "HB3"]:
            name = "HB1"


        new_atom = self.atom_lookup.get((resName, name))    
        return f'A<{new_atom}>'



def sequence_three_letter_one_hot(seq_list: list):
    """Convert protein sequence from three-letter codes to one-hot encoding"""
    # Define a dictionary to map three-letter amino acid codes to indices
    three_to_one_letter = {
        'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
        'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
        'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
        'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19,
        'UNK': 20  # 'UNK' is used for unknown or ambiguous amino acids
    }

    length = len(seq_list)
    one_hot_array = np.zeros([length, 21])

    for idx, item in enumerate(seq_list):
        item = item.upper()
        if item not in three_to_one_letter:
            item = 'UNK'
        col_idx = three_to_one_letter[item]
        one_hot_array[idx, col_idx] = 1

    # return torch.from_numpy(one_hot_array).reshape(-1, 21)
    return one_hot_array



def ss8_one_hot(ss8_list: list):
    """Convert SS8 sequence to one-hot encoding"""
    ss8_to_one_hot = {
        'H': [1, 0, 0, 0, 0, 0, 0, 0],
        'B': [0, 1, 0, 0, 0, 0, 0, 0],
        'E': [0, 0, 1, 0, 0, 0, 0, 0],
        'G': [0, 0, 0, 1, 0, 0, 0, 0],
        'I': [0, 0, 0, 0, 1, 0, 0, 0],
        'T': [0, 0, 0, 0, 0, 1, 0, 0],
        'S': [0, 0, 0, 0, 0, 0, 1, 0],
        '-': [0, 0, 0, 0, 0, 0, 0, 1]
    }
    length = len(ss8_list)
    one_hot_array = np.zeros([length, 8])

    for idx, item in enumerate(ss8_list):
        if item not in ss8_to_one_hot:
            item = '-'
        one_hot_array[idx, :] = ss8_to_one_hot[item]

    return one_hot_array



def run_dssp(pdb_file: str) -> pd.DataFrame:
    """Run biopython DSSP for SS8(3), RASA Angle(Phi, Phi)"""
    pdb_name = pdb_file.split('/')[-1].split('.')[0]
    p = PDBParser()

    structure = p.get_structure(pdb_name, pdb_file)
    model = structure[0]
    dssp = DSSP(model, pdb_file)
    key_list = list(dssp.keys())
    
    three_letter_list=[]
    desciptor_list=[]
    ss8_list = []
    rasa_list = []
    phi_list = []
    psi_list = []

    for key in key_list:
        chain_id,res_id = key
        _,resseq,icode=res_id
        residue = model[chain_id][res_id]
        residue_name = residue.get_resname()
        if icode==" ":
            desciptor=f'c<{chain_id}>r<{resseq}>R<{residue_name}>'
        else:
            desciptor=f'c<{chain_id}>r<{resseq}>i<{icode}>R<{residue_name}>'
        desciptor_list.append(desciptor)

        ss8, rasa, phi, psi = dssp[key][2:6]
        ss8_list.append(ss8)
        rasa_list.append(rasa)
        phi_list.append(phi)
        psi_list.append(psi)
        three_letter_list.append(residue_name)
    
    # Convert SS8 to one-hot encoding
    ss8_array = ss8_one_hot(ss8_list)

    # Convert one-hot arrays to DataFrame
    ss8_df = pd.DataFrame(ss8_array, columns=[f'SS8_{i}' for i in range(8)])
    one_hot_df = pd.DataFrame(sequence_three_letter_one_hot(three_letter_list),
                              columns=[f'AA_{i}' for i in range(21)])
    
    # Normalize phi and psi using MinMaxScaler
    scaler_phi = MinMaxScaler()
    phi_array = np.array(phi_list).reshape(-1, 1)
    phi_list_norm = scaler_phi.fit_transform(phi_array).flatten()

    scaler_psi = MinMaxScaler()
    psi_array = np.array(psi_list).reshape(-1, 1)
    psi_list_norm = scaler_psi.fit_transform(psi_array).flatten()

    # Concatenate one-hot DataFrames with the feature DataFrame
    feature_df = pd.DataFrame(list(zip(desciptor_list, rasa_list, phi_list_norm, psi_list_norm)),
                              columns=['ID', 'rasa', 'phi', 'psi'])
    result_df = pd.concat([feature_df, ss8_df, one_hot_df], axis=1)

    return result_df



def sequence_three_letter_one_hot(seq_list: list):
    """Convert protein sequence from three-letter codes to one-hot encoding"""
    # Define a dictionary to map three-letter amino acid codes to indices
    three_to_one_letter = {
        'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
        'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
        'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
        'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19,
        'UNK': 20  # 'UNK' is used for unknown or ambiguous amino acids
    }

    length = len(seq_list)
    one_hot_array = np.zeros([length, 21])

    for idx, item in enumerate(seq_list):
        item = item.upper()
        if item not in three_to_one_letter:
            item = 'UNK'
        col_idx = three_to_one_letter[item]
        one_hot_array[idx, col_idx] = 1

    # return torch.from_numpy(one_hot_array).reshape(-1, 21)
    return one_hot_array



def ss8_one_hot(ss8_list: list):
    """Convert SS8 sequence to one-hot encoding"""
    ss8_to_one_hot = {
        'H': [1, 0, 0, 0, 0, 0, 0, 0],
        'B': [0, 1, 0, 0, 0, 0, 0, 0],
        'E': [0, 0, 1, 0, 0, 0, 0, 0],
        'G': [0, 0, 0, 1, 0, 0, 0, 0],
        'I': [0, 0, 0, 0, 1, 0, 0, 0],
        'T': [0, 0, 0, 0, 0, 1, 0, 0],
        'S': [0, 0, 0, 0, 0, 0, 1, 0],
        '-': [0, 0, 0, 0, 0, 0, 0, 1]
    }
    length = len(ss8_list)
    one_hot_array = np.zeros([length, 8])

    for idx, item in enumerate(ss8_list):
        if item not in ss8_to_one_hot:
            item = '-'
        one_hot_array[idx, :] = ss8_to_one_hot[item]

    return one_hot_array



def run_dssp(pdb_file: str) -> pd.DataFrame:
    """Run biopython DSSP for SS8(3), RASA Angle(Phi, Phi)"""
    pdb_name = pdb_file.split('/')[-1].split('.')[0]
    p = PDBParser()

    structure = p.get_structure(pdb_name, pdb_file)
    model = structure[0]
    dssp = DSSP(model, pdb_file)
    key_list = list(dssp.keys())
    
    three_letter_list=[]
    desciptor_list=[]
    ss8_list = []
    rasa_list = []
    phi_list = []
    psi_list = []

    for key in key_list:
        chain_id,res_id = key
        _,resseq,icode=res_id
        residue = model[chain_id][res_id]
        residue_name = residue.get_resname()
        if icode==" ":
            desciptor=f'c<{chain_id}>r<{resseq}>R<{residue_name}>'
        else:
            desciptor=f'c<{chain_id}>r<{resseq}>i<{icode}>R<{residue_name}>'
        desciptor_list.append(desciptor)

        ss8, rasa, phi, psi = dssp[key][2:6]
        ss8_list.append(ss8)
        rasa_list.append(rasa)
        phi_list.append(phi)
        psi_list.append(psi)
        three_letter_list.append(residue_name)
    
    # Convert SS8 to one-hot encoding
    ss8_array = ss8_one_hot(ss8_list)

    # Convert one-hot arrays to DataFrame
    ss8_df = pd.DataFrame(ss8_array, columns=[f'SS8_{i}' for i in range(8)])
    one_hot_df = pd.DataFrame(sequence_three_letter_one_hot(three_letter_list),
                              columns=[f'AA_{i}' for i in range(21)])
    
    # Normalize phi and psi using MinMaxScaler
    scaler_phi = MinMaxScaler()
    phi_array = np.array(phi_list).reshape(-1, 1)
    phi_list_norm = scaler_phi.fit_transform(phi_array).flatten()

    scaler_psi = MinMaxScaler()
    psi_array = np.array(psi_list).reshape(-1, 1)
    psi_list_norm = scaler_psi.fit_transform(psi_array).flatten()

    # Concatenate one-hot DataFrames with the feature DataFrame
    feature_df = pd.DataFrame(list(zip(desciptor_list, rasa_list, phi_list_norm, psi_list_norm)),
                              columns=['ID', 'rasa', 'phi', 'psi'])
    result_df = pd.concat([feature_df, ss8_df, one_hot_df], axis=1)

    return result_df




def sequence_three_letter_one_hot(seq_list: list):
    """Convert protein sequence from three-letter codes to one-hot encoding"""
    # Define a dictionary to map three-letter amino acid codes to indices
    three_to_one_letter = {
        'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3, 'PHE': 4,
        'GLY': 5, 'HIS': 6, 'ILE': 7, 'LYS': 8, 'LEU': 9,
        'MET': 10, 'ASN': 11, 'PRO': 12, 'GLN': 13, 'ARG': 14,
        'SER': 15, 'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19,
        'UNK': 20  # 'UNK' is used for unknown or ambiguous amino acids
    }

    length = len(seq_list)
    one_hot_array = np.zeros([length, 21])

    for idx, item in enumerate(seq_list):
        item = item.upper()
        if item not in three_to_one_letter:
            item = 'UNK'
        col_idx = three_to_one_letter[item]
        one_hot_array[idx, col_idx] = 1

    # return torch.from_numpy(one_hot_array).reshape(-1, 21)
    return one_hot_array



def ss8_one_hot(ss8_list: list):
    """Convert SS8 sequence to one-hot encoding"""
    ss8_to_one_hot = {
        'H': [1, 0, 0, 0, 0, 0, 0, 0],
        'B': [0, 1, 0, 0, 0, 0, 0, 0],
        'E': [0, 0, 1, 0, 0, 0, 0, 0],
        'G': [0, 0, 0, 1, 0, 0, 0, 0],
        'I': [0, 0, 0, 0, 1, 0, 0, 0],
        'T': [0, 0, 0, 0, 0, 1, 0, 0],
        'S': [0, 0, 0, 0, 0, 0, 1, 0],
        '-': [0, 0, 0, 0, 0, 0, 0, 1]
    }
    length = len(ss8_list)
    one_hot_array = np.zeros([length, 8])

    for idx, item in enumerate(ss8_list):
        if item not in ss8_to_one_hot:
            item = '-'
        one_hot_array[idx, :] = ss8_to_one_hot[item]

    return one_hot_array



def run_dssp(pdb_file: str) -> pd.DataFrame:
    """Run biopython DSSP for SS8(3), RASA Angle(Phi, Phi)"""
    pdb_name = pdb_file.split('/')[-1].split('.')[0]
    p = PDBParser()

    structure = p.get_structure(pdb_name, pdb_file)
    model = structure[0]
    dssp = DSSP(model, pdb_file)
    key_list = list(dssp.keys())
    
    three_letter_list=[]
    desciptor_list=[]
    ss8_list = []
    rasa_list = []
    phi_list = []
    psi_list = []

    for key in key_list:
        chain_id,res_id = key
        _,resseq,icode=res_id
        residue = model[chain_id][res_id]
        residue_name = residue.get_resname()
        if icode==" ":
            desciptor=f'c<{chain_id}>r<{resseq}>R<{residue_name}>'
        else:
            desciptor=f'c<{chain_id}>r<{resseq}>i<{icode}>R<{residue_name}>'
        desciptor_list.append(desciptor)

        ss8, rasa, phi, psi = dssp[key][2:6]
        ss8_list.append(ss8)
        rasa_list.append(rasa)
        phi_list.append(phi)
        psi_list.append(psi)
        three_letter_list.append(residue_name)
    
    # Convert SS8 to one-hot encoding
    ss8_array = ss8_one_hot(ss8_list)

    # Convert one-hot arrays to DataFrame
    ss8_df = pd.DataFrame(ss8_array, columns=[f'SS8_{i}' for i in range(8)])
    one_hot_df = pd.DataFrame(sequence_three_letter_one_hot(three_letter_list),
                              columns=[f'AA_{i}' for i in range(21)])
    
    # Normalize phi and psi using MinMaxScaler
    scaler_phi = MinMaxScaler()
    phi_array = np.array(phi_list).reshape(-1, 1)
    phi_list_norm = scaler_phi.fit_transform(phi_array).flatten()

    scaler_psi = MinMaxScaler()
    psi_array = np.array(psi_list).reshape(-1, 1)
    psi_list_norm = scaler_psi.fit_transform(psi_array).flatten()

    # Concatenate one-hot DataFrames with the feature DataFrame
    feature_df = pd.DataFrame(list(zip(desciptor_list, rasa_list, phi_list_norm, psi_list_norm)),
                              columns=['ID', 'rasa', 'phi', 'psi'])
    result_df = pd.concat([feature_df, ss8_df, one_hot_df], axis=1)

    return result_df


def run_dssp_simple(pdb_file: str) -> pd.DataFrame:
    """get all residue"""
    pdb_name = pdb_file.split('/')[-1].split('.')[0]
    p = PDBParser()

    structure = p.get_structure(pdb_name, pdb_file)
    model = structure[0]
    dssp = DSSP(model, pdb_file)
    key_list = list(dssp.keys())
    
    desciptor_list=[]


    for key in key_list:
        chain_id,res_id = key
        _,resseq,icode=res_id
        residue = model[chain_id][res_id]
        residue_name = residue.get_resname()
        if icode==" ":
            desciptor=f'c<{chain_id}>r<{resseq}>R<{residue_name}>'
        else:
            desciptor=f'c<{chain_id}>r<{resseq}>i<{icode}>R<{residue_name}>'
        desciptor_list.append(desciptor)
    # Concatenate one-hot DataFrames with the feature DataFrame
    full_residue_df = pd.DataFrame(desciptor_list,
                              columns=['ID'])

    return full_residue_df