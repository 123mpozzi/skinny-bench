import os, re, json, time
import cv2 # to load, save, process images
import imghdr # to check if a file is an image
from random import shuffle
from math import floor

# remember that Pratheepan dataset has one file with comma in the filename
csv_sep = '?'


def get_training_and_testing_sets(file_list: list, split: float = 0.7):
    print(file_list)
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing

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

# args eg: datasets/ECU/skin_masks datasets/ECU/original_images datasets/ecu/
# note: NonDefined, TRain, TEst, VAlidation
def analyze_dataset(gt: str, ori: str, root_dir: str, note: str = 'nd',
                    gt_filename_format: str = '', ori_filename_format: str = '',
                    gt_ext: str = '', ori_ext: str = '') -> None:
    out_analysis_filename = 'data.csv'

    out_analysis = os.path.join(root_dir, out_analysis_filename)
    analyze_content(gt, ori, out_analysis, note = note,
                    gt_filename_format = gt_filename_format,
                    ori_filename_format = ori_filename_format,
                    gt_ext = gt_ext, ori_ext = ori_ext)

# creates a file with lines like: origina_image1.jpg, skin_mask1.png, tr
def analyze_content(gt: str, ori: str, outfile: str, note: str = 'nd',
                    gt_filename_format: str = '', ori_filename_format: str = '',
                    gt_ext: str = '', ori_ext: str = '') -> None:
    # images found
    i = 0

    # append to data file
    with open(outfile, 'a') as out:

        for gt_file in os.listdir(gt):
            gt_path = os.path.join(gt, gt_file)

            # controlla se e' un'immagine (per evitare problemi con files come thumbs.db)
            if not os.path.isdir(gt_path) and imghdr.what(gt_path) != None:
                matched = False
                gt_name, gt_e = os.path.splitext(gt_file)
                gt_identifier = get_variable_filename(gt_name, gt_filename_format)

                if gt_identifier == None:
                    continue

                if gt_ext and gt_e != '.' + gt_ext:
                    continue
                
                for ori_file in os.listdir(ori):
                    ori_path = os.path.join(ori, ori_file)
                    ori_name, ori_e = os.path.splitext(ori_file)
                    ori_identifier = get_variable_filename(ori_name, ori_filename_format)
                    
                    if ori_identifier == None:
                        continue

                    if ori_ext and ori_e != '.' + ori_ext:
                        continue
                    
                    # try to find a match (original image - gt)
                    if gt_identifier == ori_identifier:
                        out.write(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}\n")
                        i += 1
                        matched = True
                        break
                
                if not matched:
                    print(f'No matches found for {gt_identifier}')
            else:
                print(f'File {gt_path} is not an image')
        
        print(f"Found {i} images")

# Does a simple processing on all dataset images based on a few operations.
# Used to make ground truth masks uniform across the datasets.
# 
# (JSON) "processpipe" : "png,skin=255_255_255,invert"
#        "processout" : "out/process/folder" 
# skin=... vuol dire che la regola per binarizzare è: tutto quello che non è skin va a nero, skin a bianco
# bg=... viceversa
#    (quindi skin e bg fanno anche binarizzazione!)
# *il processing viene fatto nell'ordine scritto!!!
def process_images(data_dir: str, process_pipeline: str, out_dir = '',
                   im_filename_format: str = '', im_ext: str = '') -> str:

    # loop mask files
    for im_basename in os.listdir(data_dir):
        im_path = os.path.join(data_dir, im_basename)
        im_filename, im_e = os.path.splitext(im_basename)

        # controlla se e' un'immagine (per evitare problemi con files come thumbs.db)
        if not os.path.isdir(im_path) and imghdr.what(im_path) != None:
            if out_dir == '':
                out_dir = os.path.join(data_dir, 'processed')

            os.makedirs(out_dir, exist_ok=True)

            im_identifier = get_variable_filename(im_filename, im_filename_format)
            if im_identifier == None:
                continue
            
            if im_ext and im_e != '.' + im_ext:
                continue

            # load image
            im = cv2.imread(im_path)

            # prepare path for out image
            im_path = os.path.join(out_dir, im_basename)

            for operation in process_pipeline.split(','):
                # binarize. Rule: what isn't skin is black
                if operation.startswith('skin'):
                    # inspired from https://stackoverflow.com/a/53989391
                    bgr_data = operation.split('=')[1]
                    bgr_chs = bgr_data.split('_')
                    b = int(bgr_chs[0])
                    g = int(bgr_chs[1])
                    r = int(bgr_chs[2])
                    lower_val = (b, g, r)
                    upper_val = lower_val
                    # Threshold the image to get only selected colors
                    # what isn't skin is black
                    mask = cv2.inRange(im, lower_val, upper_val)
                    im = mask
                # binarize. Rule: what isn't bg is white
                elif operation.startswith('bg'):
                    bgr_data = operation.split('=')[1]
                    bgr_chs = bgr_data.split('_')
                    b = int(bgr_chs[0])
                    g = int(bgr_chs[1])
                    r = int(bgr_chs[2])
                    lower_val = (b, g, r)
                    upper_val = lower_val
                    # Threshold the image to get only selected colors
                    mask = cv2.inRange(im, lower_val, upper_val)
                    #cv2_imshow(mask) #debug
                    # what isn't bg is white
                    sk = cv2.bitwise_not(mask)
                    im = sk
                # invert image
                elif operation == 'invert':
                    im = cv2.bitwise_not(im)
                # convert to png
                elif operation == 'png':
                    im_path = os.path.join(out_dir, im_filename + '.png')
                # reload image
                elif operation == 'reload':
                    im = cv2.imread(im_path)
                else:
                    print(f'Image processing operation unknown: {operation}')

            # save processing 
            cv2.imwrite(im_path, im)

    return out_dir

# update the csv by adding a split from a different-format file (1 column split)
def import_split(csv_file: str, single_col_file: str, outfile: str,
                 note: str, gtf = '', orif = '', inf = '') -> None:
    # read csv lines
    file3c = open(csv_file)
    triples = file3c.read().splitlines()
    file3c.close()
    
    # read single column file lines
    file1c = open(single_col_file)
    singles = file1c.read().splitlines()
    file1c.close()

    # create the new split file as csv two columns
    with open(os.path.join(outfile), 'w') as out:
        i = 0

        for entry in triples: # oriname.ext, gtname.ext, te/tr/va
            ori_path = entry.split(csv_sep)[0]
            gt_path = entry.split(csv_sep)[1]
            note_old = entry.split(csv_sep)[2]
            ori_name, ori_ext = os.path.splitext(os.path.basename(ori_path))
            gt_name, gt_ext = os.path.splitext(os.path.basename(gt_path))

            ori_identifier = get_variable_filename(ori_name, orif)
            gt_identifier = get_variable_filename(gt_name, gtf)

            for line in singles: # imgname
                line_name, line_ext = os.path.splitext(line)
                in_identifier = get_variable_filename(line_name, inf)

                if ori_identifier == in_identifier or gt_identifier == in_identifier:
                    note_old = note
                    i += 1
                    print(f'Match found: {ori_identifier}\|{gt_identifier} - {in_identifier}')
                    break # match found
                
            out.write(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note_old}\n")
        
        print(f'''Converted {i}/{len(singles)} lines.\n
        Source file: {single_col_file}\n
        Target file: {outfile}''')

# update the notes in csv by adding a split from a partial file with the same format
def update_split(csv_file: str, partial_file: str, outfile: str, newnote: str) -> None:
    # read csv lines
    file3c = open(csv_file)
    triples = file3c.read().splitlines()
    file3c.close()
    
    # read partial file lines
    file1c = open(partial_file)
    partials = file1c.read().splitlines()
    file1c.close()

    # create the new split file as csv two columns
    with open(os.path.join(outfile), 'w') as out:
        i = 0

        for entry in triples: # oriname.ext, gtname.ext
            ori_path = entry.split(csv_sep)[0]
            gt_path = entry.split(csv_sep)[1]
            note_old = entry.split(csv_sep)[2]

            skintone = ''
            if len(entry.split(csv_sep)) == 4:
                skintone = csv_sep + entry.split(csv_sep)[3]


            for line in partials: # imgname
                ori_path_part = line.split(csv_sep)[0]

                if ori_path == ori_path_part:
                    note_old = newnote
                    i += 1
                    print(f'Match found: {ori_path}')
                    break # match found
                
            out.write(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note_old}{skintone}\n")
        
        print(f'''Updated {i}/{len(partials)} lines.\n
        Source file: {partial_file}\n
        Target file: {outfile}''')

# import dataset and generate metadata
def import_dataset(import_json: str) -> None:
    if os.path.exists(import_json):
        with open(import_json, 'r') as stream:
            data = json.load(stream)

            # load JSON values
            gt = data['gt']
            ori = data['ori']
            root = data['root']
            note = data['note']
            gt_format = data['gtf']
            ori_format = data['orif']
            gt_ext = data['gtext']
            ori_ext = data['oriext']
            ori_process = data['oriprocess']
            ori_process_out = data['oriprocessout']
            gt_process = data['gtprocess']
            gt_process_out = data['gtprocessout']
            
            # check if processing is required
            if ori_process:
                ori = process_images(ori, ori_process, ori_process_out,
                                     ori_format, ori_ext)
                # update the file extension in the images are being converted
                if 'png' in ori_process:
                    ori_ext = 'png'
            
            if gt_process:
                gt = process_images(gt, gt_process, gt_process_out,
                                     gt_format, gt_ext)
                if 'png' in gt_process:
                    gt_ext = 'png'
            
            # Non-Defined as default note
            if not note:
                note = 'nd'
            
            # analyze the dataset and create the csv files
            analyze_dataset(gt, ori, root,
                            note, gt_format, ori_format,
                            gt_ext, ori_ext)
    else:
        print("JSON import file does not exist!")

def get_timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

# from schmugge custom config (.config.SkinImManager) to a list of dict structure
def read_schmugge(skin_im_manager_path: str, images_dir: str) -> list: # also prepare the csv
    sch = []
    
    # images with gt errors, aa69 is also duplicated in the config file
    blacklist = ['aa50.gt.d3.pgm', 'aa69.gt.d3.pgm', 'dd71.gt.d3.pgm', 'hh54.gt.d3.pgm']

    with open(skin_im_manager_path) as f:
        start = 0
        i = 0
        tmp = {}
        for line in f:
            blacklisted = False

            if start < 2: # skip first 2 lines
                start += 1
                continue
            
            #print(f'{line}\t{i}') # debug
            if line: # line not empty
                line = line.rstrip() # remove End Of Line (\n)

                if i == 2: # skin tone type
                    skin_type = int(line)
                    if skin_type == 0:
                        tmp['skintone'] = 'light'
                    elif skin_type == 1:
                        tmp['skintone'] = 'medium'
                    elif skin_type == 2:
                        tmp['skintone'] = 'dark'
                    else:
                        tmp['skintone'] = 'nd'
                elif i == 3: # db type
                    tmp['db'] = line
                elif i == 8: # ori
                    tmp['ori'] = os.path.join(images_dir, line)
                elif i == 9: # gt
                    tmp['gt'] = os.path.join(images_dir, line)
                    if line in blacklist:
                        blacklisted = True
                

                # update image counter
                i += 1
                if i == 10: # 10 lines read, prepare for next image data
                    if not blacklisted:
                        sch.append(tmp)
                    tmp = {}
                    i = 0
    
    print(f'Schmugge custom config read correctly, found {len(sch)} images')

    return sch

# from schmugge list of dicts structure to csv file and processed images
def process_schmugge(sch: list, outfile: str, train = 70, test = 15, val = 15, ori_out_dir = 'new_ori', gt_out_dir = 'new_gt'):
    # prepare new ori and gt dirs
    os.makedirs(ori_out_dir, exist_ok=True)
    os.makedirs(gt_out_dir, exist_ok=True)

    with open(outfile, 'w') as out:
        shuffle(sch) # randomize

        # 70% train, 15% val, 15% test
        train_files, test_files = get_training_and_testing_sets(sch)
        test_files, val_files = get_training_and_testing_sets(test_files, split=.5)

        for entry in sch:
            db = int(entry['db'])
            ori_path = entry['ori']
            gt_path = entry['gt']
            

            ori_basename = os.path.basename(ori_path)
            gt_basename = os.path.basename(gt_path)
            ori_filename, ori_e = os.path.splitext(ori_basename)
            gt_filename, gt_e = os.path.splitext(gt_basename)

            # process images
            # load images
            ori_im = cv2.imread(ori_path)
            gt_im = cv2.imread(gt_path)
            # png
            ori_out = os.path.join(ori_out_dir, ori_filename + '.png')
            gt_out = os.path.join(gt_out_dir, gt_filename + '.png')
            # binarize gt: whatever isn't background, is skin
            if db == 4 or db == 3: # Uchile/UW: white background
                b = 255
                g = 255
                r = 255
                lower_val = (b, g, r)
                upper_val = lower_val
                # Threshold the image to get only selected colors
                mask = cv2.inRange(gt_im, lower_val, upper_val)
                #cv2_imshow(mask) #debug
                # what isn't bg is white
                sk = cv2.bitwise_not(mask)
                gt_im = sk
            else: # background = 180,180,180
                b = 180
                g = 180
                r = 180
                lower_val = (b, g, r)
                upper_val = lower_val
                # Threshold the image to get only selected colors
                mask = cv2.inRange(gt_im, lower_val, upper_val)
                #cv2_imshow(mask) #debug
                # what isn't bg is white
                sk = cv2.bitwise_not(mask)
                gt_im = sk
            # save processing 
            cv2.imwrite(ori_out, ori_im)
            cv2.imwrite(gt_out, gt_im)

            skintone = entry['skintone']
            note = 'te'
            if entry in train_files:
                note = 'tr'
            elif entry in val_files:
                note = 'va'
            
            out.write(f"{ori_out}{csv_sep}{gt_out}{csv_sep}{note}{csv_sep}{skintone}\n")

# write all csv line note attributes as the given argument
def csv_full_test(csv_file: str, note = 'nd'):
    # read the images CSV
    file = open(csv_file)
    file3c = file.read().splitlines()
    file.close()

    # rewrite csv file
    with open(csv_file, 'w') as out:
        for entry in file3c:
            ori_path = entry.split(csv_sep)[0]
            gt_path = entry.split(csv_sep)[1]
            #note = 'nd'

            # check if there is also a 4th parameter in the line (Schmugge skintones)
            skintone = ''
            if len(entry.split(csv_sep)) == 4:
                skintone = csv_sep + entry.split(csv_sep)[3]

            out.write(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}{skintone}\n")

# write all csv line note attributes as the given argument
def csv_count_test(csv_file: str, count: int, note = 'nd'):
    # read the images CSV
    file = open(csv_file)
    file3c = file.read().splitlines()
    file.close()

    # rewrite csv file
    i = 0
    with open(csv_file, 'w') as out:
        for entry in file3c:
            ori_path = entry.split(csv_sep)[0]
            gt_path = entry.split(csv_sep)[1]
            #note = 'nd'

            if i < count:
                note = 'te'
                i += 1
            else:
                note = 'tr'

            # check if there is also a 4th parameter in the line (Schmugge skintones)
            skintone = ''
            if len(entry.split(csv_sep)) == 4:
                skintone = csv_sep + entry.split(csv_sep)[3]

            out.write(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}{skintone}\n")

# all the items with note 'te' become 'tr', all the items with note != 'te', become 'te'
def csv_not_test(csv_file: str):
    # read the images CSV
    file = open(csv_file)
    file3c = file.read().splitlines()
    file.close()

    # rewrite csv file
    with open(csv_file, 'w') as out:
        for entry in file3c:
            ori_path = entry.split(csv_sep)[0]
            gt_path = entry.split(csv_sep)[1]
            nt = entry.split(csv_sep)[2]

            if nt == 'te':
                note = 'tr'
            else:
                note = 'te'

            # check if there is also a 4th parameter in the line (Schmugge skintones)
            skintone = ''
            if len(entry.split(csv_sep)) == 4:
                skintone = csv_sep + entry.split(csv_sep)[3]

            out.write(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}{skintone}\n")

# Updates the csv file by modyfing the notes of lines of the given skintone
# csv must have 4 cols!
# skintone may be: 'dark', 'light', 'medium'
def csv_skintone_filter(csv_file: str, skintone: str, mode = 'train', val_percent = .15, test_percent = .15):
    # read the images CSV
    file = open(csv_file)
    file3c = file.read().splitlines()
    file.close()

    # randomize
    shuffle(file3c)

    # calculate splits length
    totalsk = csv_skintone_count(csv_file, skintone) # total items to train/val/test on
    totalva = round(totalsk * val_percent)
    totalte = round(totalsk * test_percent)
    #totaltr = totalsk - totalva
    jva = 0
    jte = 0
    #jtr = 0


    # rewrite csv file
    with open(csv_file, 'w') as out:
        for entry in file3c:
            ori_path = entry.split(csv_sep)[0]
            gt_path = entry.split(csv_sep)[1]

            skint = entry.split(csv_sep)[3]

            if skint != skintone: # should not be filtered
                note = 'nd'
                
                out.write(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}{csv_sep}{skint}\n")
            else: # should be in the filter
                if mode == 'train': # if it is a training filter
                    if jva < totalva: # there are still places left to be in validation set
                        note = 'va'
                        jva += 1
                    elif jte < totalte: # there are still places left to be in test set
                        note = 'te'
                        jte += 1
                    else: # no more validation places to sit in, go in train set
                        note = 'tr'
                else: # if it is a testing filter, just place them all in test set
                    note = 'te'
                
                out.write(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}{csv_sep}{skintone}\n")

# Prints the total amount of items of the given skintone
def csv_skintone_count(csv_file: str, skintone: str):
    # read the images CSV
    file = open(csv_file)
    file3c = file.read().splitlines()
    file.close()

    j = 0
    # read csv file
    with open(csv_file, 'r') as out:
        for entry in file3c:
            ori_path = entry.split(csv_sep)[0]
            gt_path = entry.split(csv_sep)[1]
            note = entry.split(csv_sep)[2]
            skint = entry.split(csv_sep)[3]

            if skint == skintone:
                j += 1
                print(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}{csv_sep}{skint}")
    
    print(f"Found {j} items of type {skintone}")
    return j

# Prints the total amount of items of the given mode('train'(includes validation), 'test')
def csv_note_count(csv_file: str, mode: str):
    # read the images CSV
    file = open(csv_file)
    file3c = file.read().splitlines()
    file.close()

    j = 0
    # read csv file
    with open(csv_file, 'r') as out:
        for entry in file3c:
            ori_path = entry.split(csv_sep)[0]
            gt_path = entry.split(csv_sep)[1]
            nt = entry.split(csv_sep)[2]
            skint = entry.split(csv_sep)[3]

            notes = []
            if mode == 'train':
                notes.append("tr")
                notes.append("va")
            else:
                notes.append("te")

            if nt in notes:
                j += 1
                print(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{nt}{csv_sep}{skint}")
    
    print(f"Found {j} items of type {mode}")

# assumes the Schmugge dataset data.csv file is in dataset/Schmugge folder
# mode can either be 'train' or 'test'
def gen_sch_by_skintone(skintone: str, mode: str):
    sch_csv = 'dataset/Schmugge/data.csv'

    # re-import Schmugge
    schm = read_schmugge('dataset/Schmugge/data/.config.SkinImManager', 'dataset/Schmugge/data/data')
    process_schmugge(schm, sch_csv, ori_out_dir='dataset/Schmugge/newdata/ori', gt_out_dir='dataset/Schmugge/newdata/gt')

    csv_skintone_filter(sch_csv, skintone, mode = mode)
    csv_skintone_count(sch_csv, skintone)
    #csv_note_count(filter_by_skintone_csv, filter_mode)

# do not use with schmugge (4 columns)
def get_bench_testset(csv_file, count = 15):
    # read the images CSV
    file = open(csv_file)
    file3c = file.read().splitlines()
    file.close()

    filenames = []
    for i in range(count):
        istr = str(i).zfill(2)
        
        filenames.append(f'im000{istr}')

    #j = 0
    # rewrite csv file
    with open(csv_file, 'w') as out:
        for entry in file3c:
            ori_path = entry.split(csv_sep)[0]
            gt_path = entry.split(csv_sep)[1]
            note = 'tr'

            ori_basename = os.path.basename(ori_path)
            ori_filename, ori_ext = os.path.splitext(ori_basename)

            #if j < count:
            if ori_filename in filenames:
                note = 'te'
                
            out.write(f"{ori_path}{csv_sep}{gt_path}{csv_sep}{note}\n")
            #j+= 1
    
    #csv_note_count(csv_file, 'test')