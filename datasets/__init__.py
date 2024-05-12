from .thyroid import build as build_thyroid
from .thyroidv2 import build as build_thyroidv2
from .penn_fudan_ped import build as build_penn

def build_dataset(image_set, args):
    if args.dataset_file == 'thyroid':
        return build_thyroid(image_set, args)
    if args.dataset_file == 'thyroidv2':
        return build_thyroidv2(image_set, args)
    if args.dataset_file == 'penn':
        return build_penn(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')