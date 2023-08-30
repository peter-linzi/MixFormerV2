import argparse
import torch
import _init_paths
from lib.utils.merge import merge_template_search
# from lib.config.stark_s.config import param, update_config_from_file
# from lib.models.stark.stark_s import build_starks
from lib.utils.misc import NestedTensor
from thop import profile
from thop.utils import clever_format
import time
import importlib
from torch import nn
from lib.models.mixformer2_vit import build_mixformer2_vit_online

def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='mixformer2_vit_online', choices=['mixformer2_vit', 'mixformer2_vit_online'],
                        help='training script name')
    parser.add_argument('--config', type=str, default='288_depth8_score.yaml', help='yaml configure file name')
    parser.add_argument('--model', type=str, default='./models/mixformerv2_base.pth.tar', help="Tracking model path.")
    parser.add_argument('--onnx_path', type=str, help='onnx path to save')
    args = parser.parse_args()

    return args



def get_data(bs, t_sz, s_sz):
    template = torch.randn(bs, 3, t_sz, t_sz)
    online_template = torch.randn(bs, 3, t_sz, t_sz)
    search = torch.randn(bs, 3, s_sz, s_sz)

    return template, online_template, search


if __name__ == "__main__":
    device = "cuda:0"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    '''update param'''
    yaml_fname = 'experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.test.parameter.%s' % args.script)
    param = config_module.parameters(args.config, args.model, 3.0)

    '''set some values'''
    bs = 1
    t_sz = param.cfg.DATA.TEMPLATE.SIZE
    s_sz = param.cfg.DATA.SEARCH.SIZE

    '''import stark network module'''
    model_module = importlib.import_module('lib.models.mixformer2_vit')

    model_constructor=None
    if args.script == "mixformer2_vit":
        model_constructor = model_module.build_mixformer2_vit
    elif args.script == "mixformer2_vit_online":
        model_constructor = model_module.build_mixformer2_vit_online
    model = model_constructor(param.cfg, train=False)
    model.load_state_dict(torch.load(param.checkpoint, map_location='cpu')['net'], strict=True)
    print(f"Load checkpoint {param.checkpoint} successfully!")

    # get the template and search
    template, online_template, search = get_data(bs, t_sz, s_sz)
    # transfer to device
    model = model.to(device)
    template = template.to(device)
    online_template = online_template.to(device)
    search = search.to(device)

    # export onnx model
    input_names=['template', 'online_template', 'search']
    output_names=['pred_boxes', 'prob_l', 'prob_t', 'prob_r', 'prob_b', 'pred_scores', 'reg_tokens']
    for i in range(param.cfg.MODEL.BACKBONE.DEPTH):
        output_names.append("distill_feat_list_{}".format(i))
    torch.onnx.export(model, (template, online_template, search), args.onnx_path, opset_version=11, input_names=input_names, output_names=output_names)

