import os
import shutil
from distutils.dir_util import copy_tree


to_path = './logs/Skinny/checkpoint'

def load_checkpoint(checkpoint_filepath):
    out = None
    ext = os.path.splitext(checkpoint_filepath)[1]

    if ext == '.chkp':
        load_chkp(checkpoint_filepath)
        out = 'chkp'
    elif ext == '.pb':
        load_pb(checkpoint_filepath)
        out = 'pb'
    else:
        print(f'Unknown model filetype: {ext}')
    
    return out

def load_chkp(chkp_index_filepath):
    # clear checkpoints
    os.makedirs(to_path, exist_ok=True)
    shutil.rmtree(to_path)

    # define from/to loading directories
    from_path = os.path.join(os.path.dirname(chkp_index_filepath), '.')

    # make default model folder
    os.makedirs(to_path, exist_ok=True)

    # copy the model files
    copy_tree(from_path, to_path)


def load_pb(pb_filepath):
    # clear checkpoints
    os.makedirs(to_path, exist_ok=True)
    shutil.rmtree(to_path)

    # define from/to loading directories
    from_path = os.path.join(os.path.dirname(pb_filepath), '.')

    to_path_pb = os.path.join(to_path, 'saved_model.pb')
    # make default model folder
    os.makedirs(to_path_pb, exist_ok=True)

    # copy the model files
    copy_tree(from_path, to_path_pb)

# def load_schmugge_skintone_split(skintone):
#     !rm dataset/Schmugge/data.csv

#     if skintone == 'dark':
#         !cp drive/MyDrive/training/skinny/checkpoint-20210523-110554/dark2305_1309.csv dataset/Schmugge/data.csv
#         print(f'{skintone}(sch) split imported!')
#     elif skintone == 'medium':
#         !cp drive/MyDrive/training/skinny/checkpoint-20210523-112308/medium2305_1323.csv dataset/Schmugge/data.csv
#         print(f'{skintone}(sch) split imported!')
#     elif skintone == 'light':
#         !cp drive/MyDrive/training/skinny/checkpoint-20210523-122027/light2305_1420.csv dataset/Schmugge/data.csv
#         print(f'{skintone}(sch) split imported!')
#     else:
#         print(f'no split found for (sch) skintone: {skintone}')