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
    args = parser.parse_args()

    return args



def evaluate(model, template, online_template, search, softmax=True, run_score_head=True):
    """Compute FLOPs, Params"""
    macs, params = profile(model, inputs=(template, online_template, search, softmax, run_score_head),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs, params], "%.3f")
    print('macs is ', macs)
    print('params is ', params)

    """"Speed Test"""
    T_w = 10
    T_t = 100
    with torch.no_grad():
        for i in range(T_w):
            _ = model(template, online_template, search, softmax, run_score_head)
        start = time.time()
        for i in range(T_t):
            _ = model(template, online_template, search, softmax, run_score_head)
        end = time.time()
        print("model infer is {:0.1f}fps".format(T_t/(end-start)))




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

    # get the template and search
    template, online_template, search = get_data(bs, t_sz, s_sz)
    # transfer to device
    model = model.to(device)
    template = template.to(device)
    online_template = online_template.to(device)
    search = search.to(device)

    # evaluate the model properties
    evaluate(model, template, online_template, search, True, True)

