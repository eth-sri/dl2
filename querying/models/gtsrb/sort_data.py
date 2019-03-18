import os
import csv
import shutil

path = '../../../data/GTSRB/Final_Test/Images'
csv_path = os.path.join(path, '..', '..', '..', 'GT-final_test.csv')

for i in range(43):
    os.mkdir(os.path.join(path, "{:05d}".format(i)))

with open(csv_path) as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    iterator = iter(reader)
    next(iterator)  # skip header
    for row in iterator:
        img_path = os.path.join(path, row[0])
        destination = os.path.join(path, "{:05d}".format(int(row[-1])), row[0])
        shutil.move(img_path, destination)
