import argparse
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import torch.backends.cudnn as cudnn
from utils.tools import evaluate_predictions
from dataset.landslide_dataset import LandslideDataSet
import importlib

# Define class names and epsilon for numerical stability
CLASS_NAMES = ['Non-Landslide', 'Landslide']
EPSILON = 1e-14

def import_named_object(module_name, object_name):
    """ Import a named object from a specified module """
    try:
        module = __import__(module_name, globals(), locals(), [object_name])
    except ImportError:
        return None
    return vars(module).get(object_name)

def get_arguments():
    """ Parse command-line arguments """
    parser = argparse.ArgumentParser(description="Baseline method for Land4Seen")

    parser.add_argument("--data_dir", type=str, default='/scratch/Land4Sense_Competition_h5/',
                        help="Path to the dataset.")
    parser.add_argument("--model_module", type=str, default='model.Networks',
                        help="Module containing the model.")
    parser.add_argument("--model_name", type=str, default='unet',
                        help="Name of the model in the module.")
    parser.add_argument("--train_list", type=str, default='./dataset/train.txt',
                        help="File containing the training data list.")
    parser.add_argument("--test_list", type=str, default='./dataset/test.txt',
                        help="File containing the test data list.")
    parser.add_argument("--input_size", type=str, default='128,128',
                        help="Width and height of input images.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of images per batch.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading.")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate.")
    parser.add_argument("--num_steps", type=int, default=5000,
                        help="Total number of training steps.")
    parser.add_argument("--num_steps_stop", type=int, default=5000,
                        help="Early stopping step count.")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Weight decay for L2 regularization.")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="GPU ID for training.")
    parser.add_argument("--snapshot_dir", type=str, default='./exp/',
                        help="Directory to save model snapshots.")

    return parser.parse_args()

def main():
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # Create snapshot directory if it does not exist
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    # Parse input size
    width, height = map(int, args.input_size.split(','))
    input_size = (width, height)

    cudnn.enabled = True
    cudnn.benchmark = True

    # Load and initialize the model
    model_import = import_named_object(args.model_module, args.model_name)
    model = model_import(n_classes=args.num_classes)
    model.train()
    model = model.cuda()

    # Load training data
    train_loader = data.DataLoader(
        LandslideDataSet(
            args.data_dir, args.train_list,
            max_iters=args.num_steps_stop * args.batch_size,
            set='labeled'
        ),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True
    )

    # Load testing data
    test_loader = data.DataLoader(
        LandslideDataSet(args.data_dir, args.test_list, set='labeled'),
        batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    # Define optimizer and loss
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate, weight_decay=args.weight_decay
    )
    cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=255)
    upsample = nn.Upsample(size=(input_size[1], input_size[0]), mode='bilinear')

    hist = np.zeros((args.num_steps_stop, 3))
    best_f1_score = 0.5

    for step, batch_data in enumerate(train_loader):
        if step == args.num_steps_stop:
            break

        start_time = time.time()
        model.train()
        optimizer.zero_grad()

        images, labels, _, _ = batch_data
        images = images.cuda()
        labels = labels.cuda().long()

        predictions = model(images)
        predictions_upsampled = upsample(predictions)

        loss = cross_entropy_loss(predictions_upsampled, labels)
        loss.backward()
        optimizer.step()

        # Calculate batch accuracy
        _, predicted_labels = torch.max(predictions_upsampled, 1)
        batch_accuracy = np.sum(predicted_labels.cpu().numpy() == labels.cpu().numpy()) / len(labels.view(-1))

        hist[step, 0] = loss.item()
        hist[step, 1] = batch_accuracy
        hist[step, 2] = time.time() - start_time

        # Log training progress every 10 steps
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1}/{args.num_steps}, Time: {10 * np.mean(hist[step - 9:step + 1, 2]):.2f}s, "
                  f"Accuracy: {100 * np.mean(hist[step - 9:step + 1, 1]):.2f}%, "
                  f"Loss: {np.mean(hist[step - 9:step + 1, 0]):.4f}")

        # Evaluate model every 500 steps
        if (step + 1) % 500 == 0:
            print("Evaluating model...")
            model.eval()

            tp_all, fp_all, tn_all, fn_all = [np.zeros((args.num_classes, 1)) for _ in range(4)]
            total_valid_samples = 0
            f1_scores = np.zeros((args.num_classes, 1))

            for _, test_data in enumerate(test_loader):
                image, label, _, _ = test_data
                label = label.squeeze().numpy()
                image = image.float().cuda()

                with torch.no_grad():
                    prediction = model(image)

                _, prediction = torch.max(upsample(nn.functional.softmax(prediction, dim=1)), 1)
                prediction = prediction.squeeze().cpu().numpy()

                tp, fp, tn, fn, valid_samples = evaluate_predictions(
                    prediction.reshape(-1), label.reshape(-1), args.num_classes
                )
                tp_all += tp
                fp_all += fp
                tn_all += tn
                fn_all += fn
                total_valid_samples += valid_samples

            overall_accuracy = np.sum(tp_all) / total_valid_samples
            for i in range(args.num_classes):
                precision = tp_all[i] / (tp_all[i] + fp_all[i] + EPSILON)
                recall = tp_all[i] / (tp_all[i] + fn_all[i] + EPSILON)
                f1_scores[i] = 2 * precision * recall / (precision + recall + EPSILON)

                if i == 1:
                    print(f"===> {CLASS_NAMES[i]} Precision: {precision[0] * 100:.2f}%")
                    print(f"===> {CLASS_NAMES[i]} Recall: {recall[0] * 100:.2f}%")
                    print(f"===> {CLASS_NAMES[i]} F1 Score: {f1_scores[i][0] * 100:.2f}%")

            mean_f1 = np.mean(f1_scores)
            print(f"===> Mean F1 Score: {mean_f1 * 100:.2f}%, Overall Accuracy: {overall_accuracy * 100:.2f}%")

            if f1_scores[1] > best_f1_score:
                best_f1_score = f1_scores[1]
                model_save_path = os.path.join(args.snapshot_dir, f"step{step + 1}_f1_{int(f1_scores[1][0] * 10000)}.pth")
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    main()
