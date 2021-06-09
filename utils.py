# -*- coding: utf-8 -*-
import torch
import numpy as np
import os


def anchor_one_layer(feat_shape,
                     scales,
                     length,
                     offset=0.5):
    y = (np.arange(feat_shape).astype(np.float32) + offset) / feat_shape
    scales = np.array(scales).astype(np.float32) / length

    # ymin = np.expand_dims(y, axis=1) - scales / 2
    # ymax = np.expand_dims(y, axis=1) + scales / 2
    #
    # return np.stack([(ymin + ymax) / 2., ymax - ymin], -1)

    y = np.expand_dims(y, axis=1).repeat(len(scales), axis=1)
    scales = np.expand_dims(scales, axis=0).repeat(len(y), axis=0)
    return np.stack([y, scales], -1)


def anchors_all_layers(layers_shapes,
                       anchor_sizes,
                       length,
                       offset=0.5):
    layers_anchors = []
    for l, s in zip(layers_shapes, anchor_sizes):
        anchor_bboxes = anchor_one_layer(l,
                                         s,
                                         length,
                                         offset)
        layers_anchors.append(anchor_bboxes)
    return layers_anchors


def read_annotation(annotation_path):
    classes = sorted(os.listdir(annotation_path))
    # if os.path.exists(classes_path):
    #     os.remove(classes_path)
    # f = open(classes_path, 'w')
    annotation = {}
    class_index = {}
    tmp = []
    for index, class_name in enumerate(classes):
        class_index[class_name] = index
        if 'Ambiguous' not in class_name:
            filename = os.path.join(annotation_path, class_name)
            with open(filename, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip('\n').split()
                    if line not in tmp:
                        # if 1==1:
                        tmp.append(line)
                        video_name, start_time, end_time = line
                        if video_name not in annotation.keys():
                            annotation[video_name] = []
                        annotation[video_name].append([float(start_time), float(end_time), index])
    return annotation, class_index


def generate_sequence(data_path,
                      annotation,
                      overlap,
                      num_frame_clip,
                      fps=10,
                      min_len=4,
                      ratio=0.1,
                      threshold=0.9,
                      bi_direction=True):
    max_num = -1
    videos_name = sorted(os.listdir(data_path))
    sequence = []

    for video_name in videos_name:
        video_path = os.path.join(data_path, video_name)
        num_frames = len(os.listdir(video_path))
        num_clips = (num_frames - overlap) // (num_frame_clip - overlap)
        if num_clips > 0:
            clips = []
            for k in range(num_clips):
                start_frame = num_frame_clip * k - overlap * k
                end_frame = num_frame_clip * (k + 1) - overlap * k - 1
                assert end_frame < num_frames
                clips.append([start_frame, end_frame])

                if bi_direction:
                    start_frame = (num_frames - 1) - (num_frame_clip * (k + 1) - overlap * k - 1)
                    end_frame = (num_frames - 1) - (num_frame_clip * k - overlap * k)
                    assert end_frame < num_frames
                    clips.append([start_frame, end_frame])
        else:
            clips = [[0, num_frames - 1]]

        video_anns = annotation[video_name]
        for clip in clips:
            clip_intances = []

            start_frame_label = [0] * num_frame_clip
            end_frame_label = [0] * num_frame_clip

            start_frame, end_frame = clip
            for ann_start, ann_end, ann_label in video_anns:
                ann_start = ann_start * fps
                ann_end = ann_end * fps
                ann_len = ann_end - ann_start
                if ann_len > min_len:
                    inter_start = max(ann_start, start_frame)
                    inter_end = min(ann_end, end_frame)
                    inter_len = inter_end - inter_start
                    if (inter_len / ann_len) > threshold:
                        inter_start = inter_start - start_frame
                        inter_end = inter_end - start_frame
                        tmp_s = max(int(inter_start - ratio * inter_len), 0)
                        tmp_e = min(int(inter_start + ratio * inter_len), num_frame_clip - 1)
                        start_frame_label[tmp_s:tmp_e] = [1] * (tmp_e - tmp_s)

                        tmp_s = max(int(inter_end - ratio * inter_len), 0)
                        tmp_e = min(int(inter_end + ratio * inter_len), num_frame_clip - 1)
                        end_frame_label[tmp_s:tmp_e] = [1] * (tmp_e - tmp_s)
                        inter_start = (inter_start + 0.5) / num_frame_clip
                        inter_end = (inter_end + 0.5) / num_frame_clip
                        clip_intances.append([inter_start, inter_end, ann_label])
            if len(clip_intances) > 0:
                sequence.append(
                    {'name': video_name,
                     'clip_interval': clip,
                     'instances': clip_intances,
                     'start_label': start_frame_label,
                     'end_label': end_frame_label})
                max_num = max(max_num, len(clip_intances))
    return sequence, max_num
