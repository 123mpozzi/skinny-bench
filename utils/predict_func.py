import os
from .data_org import get_timestamp, csv_count_test, csv_full_test, csv_not_test, gen_sch_by_skintone
from .loading_func import load_checkpoint, load_schmugge_skintone_split
from .models import Skinny
from .model_func import save_x, save_y, predict_function
from ..main import levels, initial_filters, image_channels, log_dir, model_name, models
from .WorkScheduler import WorkScheduler



def cross_predict(train_db, predict_db, timestr = None, save = False):
    is_ecu = False
    if predict_db == 'ecu':
        predict_db = 'dataset/ECU'
        is_ecu = True
    elif predict_db == 'hgr':
        predict_db = 'dataset/HGR_small'
    elif predict_db == 'schmugge':
        predict_db = 'dataset/Schmugge'
    
    if timestr == None:
        timestr = get_timestamp()

    # set the whole dataset as the testing set
    whole_test = os.path.join(predict_db, 'data.csv') # dataset to process

    if is_ecu:
        csv_count_test(whole_test, 2000, 'te') # limit ecu to 2000 predictions or ram crash, do it 2 times
    else:
        csv_full_test(whole_test, 'te')

    # load model files
    chkp_ext = load_checkpoint(models[train_db])
    mod = Skinny(levels, initial_filters, image_channels, log_dir, load_checkpoint=True,
            model_name=model_name, checkpoint_extension=chkp_ext)

    # predict
    ds_name = os.path.basename(predict_db).lower()
    if ds_name == 'hgr_small':
        ds_name = 'hgr'


    out_dir = os.path.join(timestr, 'skinny', 'cross', f'{train_db}_on_{ds_name}')
    x_dir = os.path.join(out_dir, 'x')
    y_dir = os.path.join(out_dir, 'y')
    pred_dir = os.path.join(out_dir, 'p')

    scheduler = WorkScheduler()
    scheduler.add_data(None, save_x, dataset_dir = predict_db, out_dir = x_dir, preprocessor = preprocessor)
    scheduler.add_data(None, save_y, dataset_dir = predict_db, out_dir = y_dir,  preprocessor = preprocessor)
    scheduler.add_data(mod, predict_function, dataset_dir = predict_db, out_dir = pred_dir)
    scheduler.do_work()

    if is_ecu: # time to do second half
        csv_not_test(whole_test)
        scheduler = WorkScheduler()
        scheduler.add_data(None, save_x, dataset_dir = predict_db, out_dir = x_dir, preprocessor = preprocessor, skip = 2000)
        scheduler.add_data(None, save_y, dataset_dir = predict_db, out_dir = y_dir,  preprocessor = preprocessor, skip = 2000)
        scheduler.add_data(mod, predict_function, dataset_dir = predict_db, out_dir = pred_dir, skip = 2000)
        scheduler.do_work()


    # zip and save predictions
    if save:
        zip_path = 'drive/MyDrive/testing/skinny/' + timestr + '_p.zip'
        !zip -r $zip_path $timestr


def base_predict(db_name, timestr = None, save = False):
    if db_name == 'ecu':
        ds = 'dataset/ECU'
    elif db_name == 'hgr':
        ds = 'dataset/HGR_small'
    elif db_name == 'schmugge':
        ds = 'dataset/Schmugge'
    
    if timestr == None:
        timestr = get_timestamp()
    
    # load model files
    chkp_ext = load_checkpoint(models[db_name])
    mod = Skinny(levels, initial_filters, image_channels, log_dir, load_checkpoint=True,
            model_name=model_name, checkpoint_extension=chkp_ext)


    out_dir = os.path.join(timestr, 'skinny', 'base', db_name)
    x_dir = os.path.join(out_dir, 'x')
    y_dir = os.path.join(out_dir, 'y')
    pred_dir = os.path.join(out_dir, 'p')

    scheduler = WorkScheduler()
    scheduler.add_data(None, save_x, dataset_dir = ds, out_dir = x_dir, preprocessor = preprocessor)
    scheduler.add_data(None, save_y, dataset_dir = ds, out_dir = y_dir,  preprocessor = preprocessor)
    scheduler.add_data(mod, predict_function, dataset_dir = ds, out_dir = pred_dir)
    scheduler.do_work()

    # zip and save predictions
    if save:
        zip_path = 'drive/MyDrive/testing/skinny/' + timestr + '_p.zip'
        !zip -r $zip_path $timestr


def cross_predict_skintones(train_skintone, predict_skintone, timestr = None, save = False):
    db_dir = 'dataset/Schmugge'

    if timestr == None:
        timestr = get_timestamp()

    # update the csv file to set the prediction set
    gen_sch_by_skintone(predict_skintone, 'test')

    # load model files
    chkp_ext = load_checkpoint(models[train_skintone])
    mod = Skinny(levels, initial_filters, image_channels, log_dir, load_checkpoint=True,
            model_name=model_name, checkpoint_extension=chkp_ext)

    # predict

    out_dir = os.path.join(timestr, 'skinny', 'cross', f'{train_skintone}_on_{predict_skintone}')
    x_dir = os.path.join(out_dir, 'x')
    y_dir = os.path.join(out_dir, 'y')
    pred_dir = os.path.join(out_dir, 'p')

    scheduler = WorkScheduler()
    scheduler.add_data(None, save_x, dataset_dir = db_dir, out_dir = x_dir, preprocessor = preprocessor)
    scheduler.add_data(None, save_y, dataset_dir = db_dir, out_dir = y_dir,  preprocessor = preprocessor)
    scheduler.add_data(mod, predict_function, dataset_dir = db_dir, out_dir = pred_dir)
    scheduler.do_work()

    # zip and save predictions
    if save:
        zip_path = 'drive/MyDrive/testing/skinny/' + timestr + '_p.zip'
        !zip -r $zip_path $timestr

def base_predict_skintones(skintone, timestr = None, save = False):
    db_dir = 'dataset/Schmugge'

    if timestr == None:
        timestr = get_timestamp()
    
    # load skintone split
    load_schmugge_skintone_split(skintone)
    
    # load model files
    chkp_ext = load_checkpoint(models[skintone])
    mod = Skinny(levels, initial_filters, image_channels, log_dir, load_checkpoint=True,
            model_name=model_name, checkpoint_extension=chkp_ext)


    out_dir = os.path.join(timestr, 'skinny', 'base', skintone)
    x_dir = os.path.join(out_dir, 'x')
    y_dir = os.path.join(out_dir, 'y')
    pred_dir = os.path.join(out_dir, 'p')

    scheduler = WorkScheduler()
    scheduler.add_data(None, save_x, dataset_dir = db_dir, out_dir = x_dir, preprocessor = preprocessor)
    scheduler.add_data(None, save_y, dataset_dir = db_dir, out_dir = y_dir,  preprocessor = preprocessor)
    scheduler.add_data(mod, predict_function, dataset_dir = db_dir, out_dir = pred_dir)
    scheduler.do_work()

    # zip and save predictions
    if save:
        zip_path = 'drive/MyDrive/testing/skinny/' + timestr + '_p.zip'
        !zip -r $zip_path $timestr