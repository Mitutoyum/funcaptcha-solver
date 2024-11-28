# A simple script to move labels and images

# the folders must have these names (labels_split, images_spit)


import pathlib
from sys import exit as _exit
from shutil import move


path = pathlib.Path(__file__).parent / 'dataset'

labels = path / 'labels_split'
images = path / 'images_split'

if not labels.is_dir() or not images.is_dir():
    _exit('Required folders not found, exiting...')


train_percent, val_percent = .8, .2

files = list(labels.iterdir())
total = len(files)


for idx, label in enumerate(files, 1):
    image = images / f'{label.stem}.jpeg'
    dest = 'train' if idx <= int(total * train_percent) else 'val'
    if image.is_file():

        move(label.absolute(), (path / dest / 'labels').absolute())
        move(image.absolute(), (path / dest / 'images').absolute())
        print(f'Processed {label.name}')
        continue

    print(f'Image of label {label.name} not found, skipping...')

print('Splitting finished')