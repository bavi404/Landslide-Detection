import argparse
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn
from utils.tools import evaluate_predictions
from dataset.landslide_dataset import LandslideDataSet
from model.Networks import UNetModel
import h5py

# Class names for landslide detection
CLASS_NAMES = ['Non-Landslide', 'Landslide']
EPSILON = 1e-14

def import_named_object(module_name, object_name):
    """ Import a named object from a module """
    try:
        module = __import__(module_name, globals(), locals(), [object_name])
    except ImportError:
        return None
    return vars(module).get(object_name)

def get_arguments():
    """ Parse and return command-line arguments """
    parser = argparse.ArgumentParser(description="Baseline method for Land4Seen")

    parser.add_argument("--data_dir", type=str, default='/scratch/Land4Sense_Competition_h5/',
                        help="Path to the dataset.")
    parser.add_argument("--model_module", type=str, default='model.Networks',
                        help="Model module to import.")
    parser.add_argument("--model_name", type=str, default='unet',
                        help="Model name in the specified module.")
    parser.add_argument("--test_list", type=str, default='./dataset/test.txt',
                        help="File containing test image paths.")
    parser.add_argument("--input_size", type=str, default='128,128',
                        help="Width and height of input images.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for data loading.")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID for testing.")
    parser.add_argument("--snapshot_dir", type=str, default='./test_map/',
                        help="Directory to save predicted masks.")
    parser.add_argument("--restore_from", type=str, default='./exp/batch3500_F1_7396.pth',
                        help="Path to the trained model.")

    return parser.parse_args()

def main():
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Create snapshot directory if it doesn't exist
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    # Parse input size
    width, height = map(int, args.input_size.split(','))
    input_size = (width, height)

    cudnn.enabled = True
    cudnn.benchmark = True

    # Initialize the model
    model = UNetModel(num_classes=args.num_classes)

    # Load pre-trained weights
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)

    model = model.cuda()

    # Initialize test data loader
    test_loader = data.DataLoader(
        LandslideDataSet(args.data_dir, args.test_list, set='unlabeled'),
        batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # Define upsampling layer
    upsample = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')

    print("Testing started...")
    model.eval()

    for index, batch in enumerate(test_loader):
        image, _, name = batch
        image = image.float().cuda()
        name = name[0].split('.')[-2].split('/')[-1].replace('image', 'mask')
        print(f"{index + 1}/{len(test_loader)}: Testing {name}")

        with torch.no_grad():
            prediction = model(image)

        # Post-process predictions
        _, prediction = torch.max(upsample(nn.functional.softmax(prediction, dim=1)).detach(), 1)
        prediction = prediction.squeeze().cpu().numpy().astype('uint8')

        # Save prediction as HDF5
        with h5py.File(os.path.join(args.snapshot_dir, f"{name}.h5"), 'w') as hf:
            hf.create_dataset('mask', data=prediction)

if __name__ == '__main__':
    main()
