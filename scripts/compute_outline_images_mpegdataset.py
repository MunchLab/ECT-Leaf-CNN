from PIL import Image
import os
from pathlib import Path



mypath = '../data/MPEG7original_subset/'
mypath_output = '../data/MPEG7_images_small/'


done_species = []
# loop through file system
classes=[]
for path, subdirs, files in os.walk(mypath):
    classes.extend(subdirs)
    files = [f for f in files if not f[0] == '.']
    subdirs[:] = [d for d in subdirs if (d[0] != '.' and d not in done_species)]
    print('Computing image version of outlines in ', path, '...')
    print(len(files))
    for name in files:
        input_filedir = os.path.join(path, name)
        output_filedir = os.path.join(mypath_output+ input_filedir[len(mypath):-3])
        img = Image.open(input_filedir)
        #resize
        newsize = (32, 48)
        img = img.resize(newsize)

        # PNG file to save
        Path(os.path.dirname(output_filedir)).mkdir(parents=True, exist_ok=True)
        img.save(output_filedir+'png')

    print('----------------\n completed subdirectory ', path ,'\n-------------', )