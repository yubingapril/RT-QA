import os
import sys


model_file = sys.argv[1]
slice_script = sys.argv[2]
interface_dir = sys.argv[3]
slice_dir = sys.argv[4]
tmp_folder = sys.argv[5]


def cal_slice(model_file,slice_script,interface_dir,out_dir,tmp_dir):
    commend="bash "+slice_script+" "+model_file+" "+interface_dir+" "+out_dir+" "+tmp_dir
    os.system(commend)

cal_slice(model_file, slice_script, interface_dir, slice_dir, tmp_folder)
