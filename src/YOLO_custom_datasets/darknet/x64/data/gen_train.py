import os
import sys

data_type = 'train'
if len(sys.argv)>1:
    data_type = sys.argv[1]
with open(data_type + '.txt', 'w') as f:
    for root, dirs, files in os.walk(os.path.abspath("./obj/"+data_type)):
        for file in files:
            if file.endswith('.jpg'):
                pos = root.find('obj\\' + data_type)
                dir_path = root[pos:]
                f.write('data/' + os.path.join(dir_path, file).replace('\\','/') + '\n')
