import itertools
import glob
import re
from collections import Counter, defaultdict
import random
from copy import copy

query_settings = {}
query_settings['MNIST'] = {}
query_settings['MNIST']['classNN'] = ['M_NN1', 'M_NN2']
keys = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
query_settings['MNIST']['var'] = ["M_" + key for key in keys]
query_settings['MNIST']['var_to_class_var'] = list(map(str, range(10)))
query_settings['MNIST']['shape'] = ["1, 1, 28, 28"]
query_settings['MNIST']['range'] = ["[0, 1]"]
query_settings['MNIST']['img_constraint'] = ["i[:] in [0, 1]"]
query_settings['MNIST']['class'] = list(map(str, range(10)))
query_settings['MNIST']['discriminatorNN'] = ['M_D']
query_settings['MNIST']['generatorNN'] = ['M_G']
query_settings['MNIST']['adversariealInfDistance'] = ['0.3']
query_settings['MNIST']['class_not_var'] = query_settings['MNIST']['class']
query_settings['MNIST']['mask'] = ["M_mask"]
query_settings['MNIST']['not_mask'] = ["M_not_mask"]

query_settings['FASHION_MNIST'] = {}
query_settings['FASHION_MNIST']['classNN'] = ['FM_NN1', 'FM_NN2']
keys = ['top', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankleboot']
query_settings['FASHION_MNIST']['var'] = ["FM_" + key for key in keys]
query_settings['FASHION_MNIST']['var_to_class_var'] = list(map(str, range(10)))
query_settings['FASHION_MNIST']['shape'] = ["1, 1, 28, 28"]
query_settings['FASHION_MNIST']['range'] = ["[0, 1]"]
query_settings['FASHION_MNIST']['img_constraint'] = ["i[:] in [0, 1]"]
query_settings['FASHION_MNIST']['class'] = list(map(str, range(10)))
query_settings['FASHION_MNIST']['discriminatorNN'] = ['FM_D']
query_settings['FASHION_MNIST']['generatorNN'] = ['FM_G']
query_settings['FASHION_MNIST']['adversariealInfDistance'] = ['0.3']
query_settings['FASHION_MNIST']['class_not_var'] = query_settings['FASHION_MNIST']['class']
query_settings['FASHION_MNIST']['mask'] = ["FM_mask"]
query_settings['FASHION_MNIST']['not_mask'] = ["FM_not_mask"]

query_settings['CIFAR'] = {}
query_settings['CIFAR']['classNN'] = ['C_VGG', 'C_RESNET']
keys = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
query_settings['CIFAR']['var'] = ["C_" + key for key in keys]
query_settings['CIFAR']['var_to_class_var'] = list(map(str, range(10)))
query_settings['CIFAR']['shape'] = ["1, 3, 32, 32"]
query_settings['CIFAR']['range'] = ["[0, 1]"]
query_settings['CIFAR']['img_constraint'] = ["i[:] in [0, 1]"]
query_settings['CIFAR']['class'] = list(map(str, range(10)))
query_settings['CIFAR']['discriminatorNN'] = ['C_D']
query_settings['CIFAR']['generatorNN'] = ['C_G']
query_settings['CIFAR']['adversariealInfDistance'] = ['0.1']
query_settings['CIFAR']['class_not_var'] = query_settings['CIFAR']['class']
query_settings['CIFAR']['mask'] = ["C_mask"]
query_settings['CIFAR']['not_mask'] = ["C_not_mask"]

query_settings['GTSRB'] = {}
query_settings['GTSRB']['classNN'] = ['G_VGG', 'G_RESNET']
keys = ['20_speed', '30_speed', '50_speed', '60_speed', '70_speed', '80_speed', '80_lifted', '100_speed', '120_speed', 'no_overtaking_general', 'no_overtaking_trucks', 'right_of_way_crossing', 'right_of_way_general', 'give_way', 'stop', 'no_way_general', 'no_way_trucks', 'no_way_one_way', 'attention_general', 'attention_left_turn', 'attention_right_turn', 'attention_curvy', 'attention_bumpers', 'attention_slippery', 'attention_bottleneck', 'attention_construction', 'attention_traffic_light', 'attention_pedestrian', 'attention_children', 'attention_bikes', 'attention_snowflake', 'attention_deer', 'lifted_general', 'turn_right', 'turn_left', 'turn_straight', 'turn_straight_right', 'turn_straight_left', 'turn_right_down', 'turn_left_down', 'turn_circle', 'lifted_no_overtaking_general', 'lifted_no_overtaking_trucks']
query_settings['GTSRB']['var'] = ["G_" + key for key in keys]
query_settings['GTSRB']['var_to_class_var'] = list(map(str, range(43)))
query_settings['GTSRB']['shape'] = ["1, 3, 32, 32"]
query_settings['GTSRB']['range'] = ["[0, 1]"]
query_settings['GTSRB']['img_constraint'] = ["i[:] in [0, 1]"]
query_settings['GTSRB']['class'] = list(map(str, range(43)))
query_settings['GTSRB']['discriminatorNN'] = ['G_D']
query_settings['GTSRB']['generatorNN'] = ['G_G']
query_settings['GTSRB']['adversariealInfDistance'] = ['0.1']
query_settings['GTSRB']['class_not_var'] = query_settings['GTSRB']['class']
query_settings['GTSRB']['mask'] = ["G_mask"]
query_settings['GTSRB']['not_mask'] = ["G_not_mask"]


query_settings['IMAGENET'] = {}
query_settings['IMAGENET']['classNN'] = ['I_VGG16', 'I_VGG19', 'I_R50']
query_settings['IMAGENET']['var'] = ['I_manhole_cover', 'I_goldfish', 'I_guinea_pig', 'I_espresso_maker', 'I_traffic_light', 'I_koala', 'I_meerkat', 'I_teddy', 'I_radio', 'I_desktop_computer']
query_settings['IMAGENET']['var_to_class_var'] = ['640', '1', '338', '550', '920', '105', '299', '850', '754', '527']
query_settings['IMAGENET']['shape'] = ["1, 3, 224, 224"]
query_settings['IMAGENET']['range'] = ["[0, 1]"]
query_settings['IMAGENET']['img_constraint'] = ["i[:, :, :] in [0, 1]"]
query_settings['IMAGENET']['class'] = ['640', '1', '338', '550', '920', '105', '299', '850', '754', '527']
query_settings['IMAGENET']['adversariealInfDistance'] = ['0.1']
query_settings['IMAGENET']['class_not_var'] = query_settings['IMAGENET']['class']
query_settings['IMAGENET']['mask'] = ["I_mask"]
query_settings['IMAGENET']['not_mask'] = ["I_not_mask"]


skip_keys = ['class_var', 'var_not_mask']

#form https://stackoverflow.com/questions/176918/finding-the-index-of-an-item-given-a-list-containing-it-in-python
def all_indices(value, qlist):
    indices = []
    idx = -1
    while True:
        try:
            idx = qlist.index(value, idx+1)
            indices.append(idx)
        except ValueError:
            break
    return indices

def get_queries(args):
    query_files = sorted(glob.glob('./evaluation_queries/' + args.glob_pattern))
    queries = {}

    for j, query_file in enumerate(query_files):
        print(f'> {query_file}')

        with open(query_file, 'r') as f:
            lines = f.readlines()
            datasets = lines[0]
            query = "".join(lines[1:])

        datasets = datasets[3:]
        datasets = [d.strip() for d in datasets.split(',')]

        original_keys = list(set([k.replace('<<', '').replace('>>', '') for k in re.findall("<<[a-zA-Z\_]+[0-9]*>>", query)]))
        original_keys_filtered = [k for k in original_keys if k not in skip_keys]
        keys = [re.sub("[0-9]+", "", k) for k in original_keys_filtered]

        
        for d in datasets:
            if d in query_settings:
                if d not in queries:
                    queries[d] = defaultdict(list)
                s = query_settings[d]
                val_sets = [s[k] for k in keys]
                val_sets = list(itertools.product(*val_sets))

                random.shuffle(val_sets)
                for vals in val_sets:
                    if 'class_var' in original_keys:
                        class_var = s['var_to_class_var'][keys.index('var')]
                    if 'class_var' in original_keys and 'class_not_var' in original_keys and vals[keys.index('class_not_var')] == class_var:
                        #print('skipping due to class_not_var')
                        continue

                    skip = False
                    for v, c in Counter(keys).items():
                        if c > 1:
                            indices = all_indices(v, keys)
                            v_check = [vals[i] for i in indices]
                            if len(v_check) != len(set(v_check)):
                                skip = True
                                break

                    if skip:
                        #print('skipping due to value overlap')
                        continue

                    query_text = copy(query)

                    for i, k in enumerate(original_keys_filtered):
                        query_text = query_text.replace(f"<<{k}>>", vals[i])
                    if 'class_var' in original_keys:
                        query_text = query_text.replace('<<class_var>>', class_var)

                    if 'var_not_mask' in original_keys:
                        var = vals[keys.index("var")]
                        query_text = query_text.replace('<<var_not_mask>>', var + "_not_mask")

                    queries[d][query_file].append(query_text)
                    if len(queries[d][query_file]) == args.instances:
                        break
    return queries
