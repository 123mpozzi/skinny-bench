import imghdr
import json
import os
import re
import time
import traceback

import cv2

# remember that Pratheepan dataset has one file with comma in the filename
csv_sep = '?'


def get_timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def read_csv(csv_file) -> list:
    '''Return the multi-column content of the dataset CSV file'''
    file_content = None
    try: #  may fail on accessing CSV file: eg. file does not exist
        file = open(csv_file)
        file_content = file.read().splitlines() #  multi-column file
        file.close()
    except Exception:
        print(traceback.format_exc())
        print('Error on accessing ' + csv_file)
        exit()
    return file_content

def split_csv_fields(row: str) -> list:
    return row.split(csv_sep)

def to_csv_row(*args) -> str:
    row = args[0]
    for item in args[1:]:
        row += csv_sep
        row += item
    row += '\n'
    return row

def match_rows(csv_file: str, targets: list, target_column: int) -> list:
    '''Return all rows matching targets in the given column'''
    csv_content = read_csv(csv_file)
    data = [x for x in csv_content if split_csv_fields(x)[target_column] in targets]
    return data

# Get the variable part of a filename into a dataset
def get_variable_filename(filename: str, format: str) -> str:
    if format == '':
        return filename

    # re.fullmatch(r'^img(.*)$', 'imgED (1)').group(1)
    # re.fullmatch(r'^(.*)-m$', 'att-massu.jpg-m').group(1)
    match =  re.fullmatch('^{}$'.format(format), filename)
    if match:
        return match.group(1)
    else:
        #print('Cannot match {} with pattern {}'.format(filename, format))
        return None

def is_image(path: str) -> bool:
    return os.path.isfile(path) and imghdr.what(path) != None


def analyze_dataset(gt: str, ori: str, root_dir: str, note: str = 'nd',
                    gt_filename_format: str = '', ori_filename_format: str = '') -> None:
    '''Create CSV file containing dataset metadata (such as paths of images)'''
    out_analysis_filename = 'data.csv'
    out_file = os.path.join(root_dir, out_analysis_filename)
    
    #  Number of images found
    i = 0

    #  Append to data file
    with open(out_file, 'a') as out:
        for gt_file in os.listdir(gt):
            gt_path = os.path.join(gt, gt_file)

            #  Check if current file is an image (avoid issues with files like thumbs.db)
            if is_image(gt_path):
                matched = False
                gt_name, gt_e = os.path.splitext(gt_file)
                gt_identifier = get_variable_filename(gt_name, gt_filename_format)

                if gt_identifier == None:
                    continue
                
                for ori_file in os.listdir(ori):
                    ori_path = os.path.join(ori, ori_file)
                    ori_name, ori_e = os.path.splitext(ori_file)
                    ori_identifier = get_variable_filename(ori_name, ori_filename_format)
                    
                    if ori_identifier == None:
                        continue
                    
                    #  Try to find a match (original image - gt)
                    if gt_identifier == ori_identifier:
                        out.write(to_csv_row(ori_path, gt_path, note))
                        i += 1
                        matched = True
                        break
                
                if not matched:
                    print(f'No matches found for {gt_identifier}')
            else:
                print(f'File {gt_path} is not an image')
        
        print(f"Found {i} images")

# Perform image-processing on a directory content
# 
# Processing Pipeline example:
#   "png,skin=255_255_255,invert"
#   skin=.. Skin-based binarization rule:
#           pixels of whatever is not skin will be set black; skin pixels will be set white
#   bg=..   Background-based binarization rule:
#           pixels of whatever is not background will be set white; background pixels will be set black
#   png     Convert the image to PNG format
# Processing operations are performed in order!
def process_images(data_dir: str, process_pipeline: str, out_dir = '', im_filename_format: str = '') -> str:
    #  Loop all files in the directory
    for im_basename in os.listdir(data_dir):
        im_path = os.path.join(data_dir, im_basename)
        im_filename, im_e = os.path.splitext(im_basename)

        #  Check if current file is an image (avoid issues with files like thumbs.db)
        if is_image(im_path):
            if out_dir == '':
                out_dir = os.path.join(data_dir, 'processed')

            os.makedirs(out_dir, exist_ok=True)

            im_identifier = get_variable_filename(im_filename, im_filename_format)
            if im_identifier == None:
                continue

            #  Load image
            im = cv2.imread(im_path)

            #  Prepare path for out image
            im_path = os.path.join(out_dir, im_basename)

            for operation in process_pipeline.split(','):
                #  Binarize
                if operation.startswith('skin') or operation.startswith('bg'):
                    #  inspired from https://stackoverflow.com/a/53989391
                    bgr_data = operation.split('=')[1]
                    b,g,r = [int(i) for i in bgr_data.split('_')]
                    lower_val = (b, g, r)
                    upper_val = lower_val

                    #  If 'skin': catch only skin pixels via thresholding
                    #  If 'bg':   catch only background pixels via thresholding
                    mask = cv2.inRange(im, lower_val, upper_val)
                    im = mask if operation.startswith('skin') else cv2.bitwise_not(mask)
                #  Invert image
                elif operation == 'invert':
                    im = cv2.bitwise_not(im)
                #  Convert to png
                elif operation == 'png':
                    im_path = os.path.join(out_dir, im_filename + '.png')
                #  Reload image
                elif operation == 'reload':
                    im = cv2.imread(im_path)
                else:
                    print(f'Image processing operation unknown: {operation}')
            
            #  Save processing 
            cv2.imwrite(im_path, im)
    return out_dir

def import_dataset(import_json: str) -> None:
    '''Import dataset and generate metadata'''
    if os.path.exists(import_json):
        with open(import_json, 'r') as stream:
            data = json.load(stream)

            #  Load JSON values
            gt = data['gt']
            ori = data['ori']
            root = data['root']
            note = data['note']
            gt_format = data['gtf']
            ori_format = data['orif']
            ori_process = data['oriprocess']
            ori_process_out = data['oriprocessout']
            gt_process = data['gtprocess']
            gt_process_out = data['gtprocessout']
            
            #  Non-Defined as default note
            if not note:
                note = 'nd'
            
            #  Check if processing is required
            if ori_process:
                ori = process_images(ori, ori_process, ori_process_out, ori_format)
            if gt_process:
                gt = process_images(gt, gt_process, gt_process_out, gt_format)
            
            #  Analyze the dataset and create the csv files
            analyze_dataset(gt, ori, root,
                            note, gt_format, ori_format)
    else:
        print("JSON import file does not exist!")

def csv_note_count(csv_file: str, mode: str):
    '''Print the total amount of items of the given mode; "train" mode also includes validation'''
    targets = ('tr', 'va') if mode == 'train' else ('te')
    data = match_rows(csv_file, (targets), 2)
    data_len = len(data)
    # print('\n'.join(data))  #  debug
    print(f"Found {data_len} items of type {' '.join(targets)}")
    return data_len

def get_bench_testset(csv_file, stop = 15):
    file_content = read_csv(csv_file)

    filenames = []
    for i in range(stop):
        # Note: filename format may differ in other versions
        istr = str(i).zfill(5)
        filenames.append(f'im{istr}')

    with open(csv_file, 'w') as out:
        for entry in file_content:
            csv_fields = split_csv_fields(entry)
            ori_path = csv_fields[0]
            gt_path = csv_fields[1]
            note = 'tr'

            ori_basename = os.path.basename(ori_path)
            ori_filename = os.path.splitext(ori_basename)[0]

            if ori_filename in filenames:
                note = 'te'
            
            out.write(to_csv_row(ori_path, gt_path, note))
    csv_note_count(csv_file, 'test') # debug
