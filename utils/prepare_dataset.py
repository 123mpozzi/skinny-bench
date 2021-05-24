from .data_org import import_dataset, read_schmugge, process_schmugge

# generate datasets metadata

# simple datasets
import_dataset("dataset/import_ecu.json")

# hgr is composed of 3 sub datasets
import_dataset("dataset/import_hgr1.json")
import_dataset("dataset/import_hgr2a.json")
import_dataset("dataset/import_hgr2b.json")

# schmugge dataset has really different filename formats but has a custom config file included
schm = read_schmugge('dataset/Schmugge/data/.config.SkinImManager', 'dataset/Schmugge/data/data')
process_schmugge(schm, 'dataset/Schmugge/data.csv', ori_out_dir='dataset/Schmugge/newdata/ori', gt_out_dir='dataset/Schmugge/newdata/gt')


# TODO: imprt splits, manually