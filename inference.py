import os 
from src.get_interface import interface_batch
from src.get_voronoi_area_order1 import batch_cal_area_order1
from src.get_graph_order1 import batch_graph_order1
from src.get_graph_order2 import batch_graph_order2
from src.get_graph_order3 import batch_graph_order3
from src.data import get_loader_three
from src.RT_GAT import RT_GAT
from argparse import ArgumentParser
from joblib import Parallel,delayed
import subprocess
import pandas as pd 
import pytorch_lightning as pl
import torch 
import json
from torch_scatter import scatter_mean


BATCH_SIZE=1
n_jobs=1
parser = ArgumentParser(description='Evaluate protein complex structures')
parser.add_argument('--complex_folder', '-c', type=str, required=True,default='./pdb')
parser.add_argument('--rt_folder', '-rt', type=str, required=True,default='./rt')
parser.add_argument('--work_dir', '-w', type=str, help='working director to save temporary files', required=True,default='./work')
parser.add_argument('--result_folder', '-r', type=str, help='The ranking result', required=True,default='./result')
parser.add_argument('--delete_tmp', '-s', type=bool, help='Save working director or not', default=False, required=False)
args = parser.parse_args()

complex_folder = args.complex_folder
rt_folder = args.rt_folder
work_dir = args.work_dir
result_folder = args.result_folder
delete_tmp = args.delete_tmp
current_folder = os.getcwd()
tmp_folder = os.path.join(current_folder,'tmp')

if not os.path.isdir(complex_folder):
    raise FileNotFoundError(f'Please check complex folder {complex_folder}')
else:
    complex_folder = os.path.abspath(complex_folder)

if len(os.listdir(complex_folder)) == 0:
    raise ValueError(f'The complex folder is empty.')

if not os.path.isdir(work_dir):
    print(f'Creating work folder')
    os.makedirs(work_dir)

if not os.path.isdir(result_folder):
    print(f'Creating result folder')
    os.makedirs(result_folder)

if not os.path.isdir(tmp_folder):
    print(f'Creating result folder')
    os.makedirs(tmp_folder)

work_dir = os.path.abspath(work_dir)
result_folder = os.path.abspath(result_folder)
print(work_dir)

interface_dir = os.path.join(work_dir,'interface')
slice_dir=os.path.join(work_dir,'slice')
contact_dir=os.path.join(work_dir,'contact')
order1_graph_dir=os.path.join(work_dir,'order1_graph')
order2_graph_dir=os.path.join(work_dir,'order2_graph')
order3_graph_dir=os.path.join(work_dir,'order3_graph')
os.makedirs(interface_dir, exist_ok=True)
os.makedirs(slice_dir, exist_ok=True)
os.makedirs(contact_dir, exist_ok=True)
os.makedirs(order1_graph_dir, exist_ok=True)
os.makedirs(order2_graph_dir, exist_ok=True)
os.makedirs(order3_graph_dir, exist_ok=True)

#### get interface of protein complex 
interface_batch(complex_folder,interface_dir,n_jobs)


#### Compute Voronoi tessellation
slice_script=os.path.join(current_folder,'src','cal_slice.sh')
slice_py=os.path.join(current_folder,'src','cal_slice.py')
model_list_all=os.listdir(interface_dir)
os.chdir(rt_folder)
# Parallel(n_jobs=n_jobs)(delayed(cal_slice)(model_file,slice_script,interface_dir,slice_dir,tmp_folder) for model_file in model_list_all)
def call_slice_in_env(model_file, slice_script, interface_dir, slice_dir, tmp_folder,slice_py):
    bash_cmd = f"""
    source /backup/data6/hanbingqing/anaconda3/etc/profile.d/conda.sh
    conda activate rt
    python {slice_py} {model_file} {slice_script} {interface_dir} {slice_dir} {tmp_folder}
    """
    subprocess.run(["bash", "-c", bash_cmd], check=True)

    
Parallel(n_jobs=n_jobs)(
    delayed(call_slice_in_env)(model_file, slice_script, interface_dir, slice_dir, tmp_folder,slice_py)
    for model_file in model_list_all
)



##### calculate voronoi contact area 
os.chdir(current_folder)
model_list=[file.split('.')[0] for file in os.listdir(slice_dir)]
batch_cal_area_order1(model_list,interface_dir,slice_dir,contact_dir,n_jobs)





## generate Delaunay graph representation of different orders
 
# Load precomputed potential-related files 
potential_dir = os.path.join(current_folder,'potential_file')

energy_file = os.path.join(potential_dir,'energy_re.csv')
energy_df=pd.read_csv(energy_file)

energy_file_mock = os.path.join(potential_dir,'energy_re_mock.csv')
energy_df_mock=pd.read_csv(energy_file_mock)

energy_file_order2 = os.path.join(potential_dir,'energy_re_order2.csv')
energy_df_order2 = pd.read_csv(energy_file_order2)
# print(energy_df_order2)

atom_file=os.path.join(potential_dir,'atom_type.txt')
atom_df=pd.read_csv(atom_file,sep='\s+',header=None,names=['res','atom','new_atom'])
atom_lookup = { 
    (row['res'], row['atom']): row['new_atom'] 
    for _, row in atom_df.iterrows() 
}


# generate order-1 graph
batch_graph_order1(complex_folder,contact_dir,order1_graph_dir,energy_df,energy_df_mock,atom_lookup,n_jobs)

# generate order-2 graph 
batch_graph_order2(complex_folder,contact_dir,slice_dir,interface_dir,order2_graph_dir,energy_df,energy_df_mock,energy_df_order2,atom_lookup,n_jobs)
print('complete order2')

# generate order-3 graph 
batch_graph_order3(slice_dir,interface_dir,energy_df,order3_graph_dir,contact_dir,complex_folder,n_jobs,dataname='test')
print('complete order3')


###### predict quality score by RT-QA model 
# load data 
g_list1=[file.split('.')[0] for file in os.listdir(order1_graph_dir)]
g_list2=[file.split('.')[0] for file in os.listdir(order2_graph_dir)]
g_list3=[file.split('.')[0] for file in os.listdir(order3_graph_dir)]
model_list = list(set(g_list1) & set(g_list2) & set(g_list3))
eval_loader = get_loader_three(order1_graph_dir,order2_graph_dir,order3_graph_dir,model_list,BATCH_SIZE)
print('complete data loader')

# load parameters 
config_path = os.path.join(current_folder,'config','rt_config.json')
with open(config_path, "r") as f:
    config = json.load(f)

# load model 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt_file = os.path.join(current_folder,'model','rtqa.ckpt')
checkpoint = torch.load(ckpt_file, map_location=device)
model = RT_GAT(pooling_type=config['pooling_type'],heads=config['heads'])
model.load_state_dict(checkpoint['state_dict'])
model.eval() # turn on model eval mode 

# predict 
pred_dockq=[]
for idx,batch_graphs in enumerate(eval_loader):
    g_batch_1,g_batch_2,g_batch_3 = batch_graphs[0],batch_graphs[1],batch_graphs[2]
    batch_scores,node_scores = model.forward(g_batch_1,g_batch_2,g_batch_3)
    
    # graph-level score: [batch_size]
    batch_scores = batch_scores.cpu()

    # node-level score
    # node_avg = node_scores.mean(dim=1).cpu()
    node_avg = scatter_mean(node_scores, g_batch_2.batch, dim=0).cpu()

    
    # combine two scores 
    fused_scores = (batch_scores + node_avg)/2
    
    
    pred_dockq.extend(fused_scores.cpu().data.numpy().tolist())
pred_dockq = [i[0] for i in pred_dockq]
df = pd.DataFrame(list(zip(model_list, pred_dockq)), columns=['MODEL', 'PRED_DOCKQ'])
model_list_all = [file.split('.')[0] for file in os.listdir(order2_graph_dir)]
missing_models = set(model_list_all) - set(model_list)

# If no interface is found between chains (e.g., due to large spatial separation),
# the model is considered invalid, and a predicted DockQ score of 0.0 is assigned.
df_missing = pd.DataFrame({'MODEL':list(missing_models), 'PRED_DOCKQ':0.0})
df_full = pd.concat([df, df_missing], ignore_index=True)

df_full.to_csv(os.path.join(result_folder, 'result.csv'), index=False)


