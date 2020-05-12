# eval_voc mAP
from __future__ import print_function
import argparse
import xml.etree.ElementTree as ET
import os,sys
import pickle
import numpy as np

def parse_args():
    """ Parse Input arguments """
    parser = argparse.ArgumentParser(description='mAP Caculation')

    parser.add_argument('--path', dest='path', help='The data path', type=str)
    args = parser.parse_args()

    return args

def parse_rec(filename):
    """ parse a pascal VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text 
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text),
                              int(bbox.find('ymin').text),
                              int(bbox.find('xmax').text),
                              int(bbox.find('ymax').text)]
        objects.append(obj_struct)
    return objects

def _do_python_eval(res_prefix, output_dir = 'output'):
    _devkit_path = '/data/BDD100Kdevkit'
    _year = '2007'
    _classes = ('__background__','car', 'person','rider')
    res_prefix = res_prefix + 'comp4_det_test_'
    filename = res_prefix + '{:s}.txt'
    annopath = os.path.join(
        _devkit_path,
        'BDD100K',
        'Annotation',
        'val',
        '{:s}.xml'
    )
    imagesetfile = os.path.join(
        _devkit_path,
        "BDD100K",
        'ImageSets',
        'Main',
        'test.txt'
    )
    cachedir = os.path.join(_devkit_path, 'annotations_cache')
    aps = []
    # The PASCAL VOC metric changed in 2010
    # use_07_metric = True if int(_year) < 2010 else False
    use_07_metric = False
    print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    for i, cls in enumerate(_classes):
        if cls == '__backgroud__':
            continue

        rec, prec, ap = voc_eval(
            filenamem, annopath, imagesetfile,cls,cachedir,ovthres=0.5,
            use_07_metric = use_07_metric)
        aps += [ap]
        print('AP for {} = {:.4f}'.format(cls, ap))
        with open(os.path.join(output_dir,cls+'_pr.pkl'),'wb') as f:
            pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
    print('MEAN AP = {:.4f}'.format(np.mean(aps)))
    print('~~~~~~~~~~~')
    print('Results:')
    for ap in aps:
        print('{:.3f}'.format(ap))
    print('{:.3f}'.format(np.mean(aps)))
    print('~~~~~~~~~~~~')



if __name__ == '__main__':
    args = parse_args()
    _do_python_eval(args.path, output_dir = 'output')


