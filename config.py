import argparse
from collections import namedtuple

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', type=str, default='./dataset/img_fps/')
parser.add_argument('--train_annotation_path', type=str, default='./annotation/train')
parser.add_argument('--val_data_path', type=str, default='./dataset/test_fps/')
parser.add_argument('--val_annotation_path', type=str, default='./annotation/test')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--epoch', type=int, default=20)
parser.add_argument('--num_frame_clip', type=int, default=256)
parser.add_argument('--max_instance', type=int, default=8)
parser.add_argument('--train_overlap', type=int, default=224)
parser.add_argument('--val_overlap', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--num_classes', type=int, default=21)
parser.add_argument('--crop_size', type=int, default=96)
parser.add_argument('--fps', type=float, default=10)
parser.add_argument('--sum_freq', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=3000)
parser.add_argument('--save_path', type=str, default='save_thumos')
cfg = parser.parse_args()

NetParams = namedtuple('Parameters', ['anchor_sizes', 'layer_shape', 'milestones', 'weights'])

params = NetParams(anchor_sizes=[[8, 15], [16, 24, 32], [32, 50], [64, 84], [128, 160], [200, 256]],
                   layer_shape=[32, 16, 8, 4, 2, 1],
                   milestones=[100000, ],
                   weights={
                       'cpd_pos': 1,
                       'cpd_neg': 1,
                       'cpd_loc': 1,
                       'rpd_pos': 1,
                       'rpd_neg': 1,
                       'rpd_loc': 1,
                       'fgd_bsn': 1,
                       'fgd_loc': 1})
