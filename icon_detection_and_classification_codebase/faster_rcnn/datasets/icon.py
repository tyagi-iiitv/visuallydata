# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
import os.path as osp
import PIL
import numpy as np
import scipy.sparse
import xml.etree.ElementTree as ET
import cPickle
import random

from .imdb import imdb
from ..utils.cython_bbox import bbox_overlaps
from .voc_eval import voc_eval

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from ..fast_rcnn.config import cfg

# <<<< obsolete

ROOT_DIR = osp.join(osp.dirname(__file__), '..', '..')
MATLAB = 'matlab_r2013b'


class icon(imdb):
    """Image database."""

    def __init__(self, image_set, db_name=None):
        imdb.__init__(self, 'icon_' + image_set)
        self._db_name = db_name
        self._id = random.random()
        self._image_set = image_set
        self._image_ext = '.png'
        self._data_path = self._get_default_path()
        self._classes = cfg.CLASSES
        print('num_classes:' + str(self.num_classes))
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_index = self._load_image_set_index()
        self._roidb_handler = self.gt_roidb
        self._num_boxes_proposal = 0

        # Use this dict for storing dataset specific config options
        self.config = {}

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        image_set_file = os.path.join(self._data_path, 'ImageSets',
                                      self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index
    
    def gt_roidb(self):
        """
        Return the database of regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path,
                                  self.name + '_region_proposal_roidb.pkl')
#        if os.path.exists(cache_file):
#            with open(cache_file, 'rb') as fid:
#                roidb = cPickle.load(fid)
#            print '{} roidb loaded from {}'.format(self.name, cache_file)
#            return roidb

        print 'Loading region proposal network boxes...'
        roidb = self._load_rpn_roidb(None)
        print 'Region proposal network boxes loaded'
        print '{} region proposals per image'.format(self._num_boxes_proposal / len(self.image_index))

        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote roidb to {}'.format(cache_file)

        return roidb

    def _get_default_path(self):
        """
        Return the default path where ICON is expected to be installed.
        """
        if self._db_name is not None:
            return os.path.join(ROOT_DIR, 'data','ICON_'+self._db_name)
        else:
            return os.path.join(ROOT_DIR, 'data', 'ICON')
    
    def image_path_at(self, i):
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier
        :param index filename stem e.g. 000000
        :return filepath
        """
        image_path = os.path.join(self._data_path, 'Images', self._image_set, index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path
    
    def _load_rpn_roidb(self, gt_roidb):
        box_list = []
        class_list = []
        for index in self.image_index:
            filename = os.path.join(self._data_path, 'Annotations', self._image_set, index + '.xml')
            assert os.path.exists(filename), \
                'RPN data not found at: {}'.format(filename)
                
            tree = ET.parse(filename)
            objs = tree.findall('object')
            num_objs = len(objs)
            
            bboxes = np.zeros((num_objs, 4), dtype=np.int32)
            classes = np.zeros(num_objs, dtype=np.int32)
            for i, obj in enumerate(objs):
                bbox = obj.find('bbox')
                x1 = float(bbox.find('x1').text)
                y1 = float(bbox.find('y1').text)
                x2 = float(bbox.find('x2').text)
                y2 = float(bbox.find('y2').text)
                bboxes[i, :] = [x1, y1, x2, y2]
                cls = obj.find('class').text
                classes[i] = self._class_to_ind[cls]

            self._num_boxes_proposal += num_objs
            box_list.append(bboxes)
            class_list.append(classes)

        return self.create_roidb_from_box_list(box_list, class_list, gt_roidb)
        
    def create_roidb_from_box_list(self, box_list, class_list, gt_roidb):
        assert len(box_list) == self.num_images, \
            'Number of boxes must match number of ground-truth images'
        roidb = []
        for i in xrange(self.num_images):
            boxes = box_list[i]
            gt_classes = class_list[i]
            num_boxes = boxes.shape[0]
            overlaps = np.zeros((num_boxes, self.num_classes), dtype=np.float32)

            if gt_roidb is not None and gt_roidb[i]['boxes'].size > 0:
                gt_boxes = gt_roidb[i]['boxes']
                gt_classes = gt_roidb[i]['gt_classes']
                gt_overlaps = bbox_overlaps(boxes.astype(np.float),
                                            gt_boxes.astype(np.float))
                argmaxes = gt_overlaps.argmax(axis=1)
                maxes = gt_overlaps.max(axis=1)
                I = np.where(maxes > 0)[0]
                overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]

            overlaps = scipy.sparse.csr_matrix(overlaps)
            roidb.append({
                'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': np.zeros((num_boxes,), dtype=np.float32),
            })
        return roidb
    
    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(self.classes):
            if cls == '__background__':
                continue
            print 'Writing {} VOC results file'.format(cls)
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))
                        
    def _get_voc_results_file_template(self):
        filename = '_ICON_' + self._image_set + '_' + str(self._id) + '_{:s}.txt'
        dat_name = 'ICON_' + self._db_name if self._db_name is not None else 'ICON'
        filedir = os.path.join(self._get_default_path(), 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path
                        
    def evaluate_detections(self, all_boxes, output_dir):
        self._write_voc_results_file(all_boxes)
        annopath = os.path.join(
                self._get_default_path(),
                'Annotations',
                self._image_set,
                '{:s}.xml')
        imagesetfile = os.path.join(
            self._get_default_path(),
            'ImageSets',
            self._image_set + '.txt')
        cachedir = os.path.join(self._get_default_path(), 'annotations_cache')
        aps = []
        recs = []
        precs = []
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            filename = self._get_voc_results_file_template().format(cls)
            rec, prec, ap = voc_eval(
                filename, annopath, imagesetfile, cls, cachedir, ovthresh=0.5)
            aps += [ap]
            recs += [rec]
            precs += [prec]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'w') as f:
                cPickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        return np.mean(aps), np.mean(recs), np.mean(precs) 

