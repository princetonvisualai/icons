"""This file is for calculating influence score over validation set"""
import argparse
import os
import torch

def calculate_influence_score(training_info: torch.Tensor, validation_info: torch.Tensor):
    """Calculate the influence score.

    Args:
        training_info (torch.Tensor): training info (gradients/representations) stored in a tensor of shape N x N_DIM
        validation_info (torch.Tensor): validation info (gradients/representations) stored in a tensor of shape N_VALID x N_DIM
    """
    # N x N_VALID
    influence_scores = torch.matmul(
        training_info, validation_info.transpose(0, 1))
    return influence_scores


def main(args):
    device = 'cpu' if not torch.cuda.is_available() else 'cuda'
    try:
        training_info = torch.load(args.train_gradient_path)
        if not torch.is_tensor(training_info):
            training_info = torch.tensor(training_info)
        training_info = training_info.to(device).float()
    except Exception as e:
        print(f"Error loading training info: {e}")
        return
    try:
        validation_path = args.validation_gradient_path
        if os.path.isdir(validation_path):
            validation_path = os.path.join(validation_path, "all_normalized.pt")
        validation_info = torch.load(validation_path)
        if not torch.is_tensor(validation_info):
            validation_info = torch.tensor(validation_info)
        validation_info = validation_info.to(device).float()
    except Exception as e:
        print(f"Error loading validation info: {e}")
        return

    influence_score = calculate_influence_score(training_info=training_info, validation_info=validation_info)
    print("The shape of influence score is: ", influence_score.shape)

    influence_output_file = os.path.join(args.influence_score, f"{args.train_file_name}_influence_score.pt")
    torch.save(influence_score, influence_output_file)
    print(f"Saved influence score to {influence_output_file}")

    output_file_txt = os.path.join(args.influence_score, f"{args.train_file_name}_influence_score.txt")
    with open(output_file_txt, 'w') as f:
        f.write(f"Influence score shape: {influence_score.shape}\n\n")
        f.write(str(influence_score))
    print(f"Saved influence score details to {output_file_txt}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description='Script for selecting the data for training')
    argparser.add_argument('--train_gradient_path', type=str, 
                        help='The path to the gradient file')
    argparser.add_argument('--validation_gradient_path', type=str, 
                        help='The path to the validation file')
    argparser.add_argument('--train_file_name', type=str, 
                        help='The name of the training file')
    argparser.add_argument('--ckpts', type=int, nargs='+',
                        help="Checkpoint numbers.")
    argparser.add_argument('--checkpoint_weights', type=float, nargs='+',
                        help="checkpoint weights")
    argparser.add_argument('--influence_score', type=str, default="selected_data",
                        help='The path to the output')
    args = argparser.parse_args()
    main(args)
