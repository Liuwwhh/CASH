import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import math
import argparse
from pathlib import Path
from model import HashingModel
from train import train_model
from load_data import get_loader
from utils import logger, set_seed, creat_result_dict, save_result_dict, InfoNCELoss

parser = argparse.ArgumentParser()
# runid
parser.add_argument('--runid', type=str, default='996', help='run id')

# path
parser.add_argument("--data_path", default="/path/to/data/", type=str)
parser.add_argument("--output_dir", default="./outputs/", type=str)

# training parameters
parser.add_argument("--epoch", default=100, type=int)
parser.add_argument('--valid', type=bool, default=True, help='Whether to valid after per train.')
parser.add_argument('--valid_epoch', type=int, default=1, help='Number of epochs to valid.')
parser.add_argument('--seed', type=int, default=88)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--feature_dim', type=int, default=512, help='dim of feature.')
parser.add_argument('--num_tasks', type=int, default=5, help='number of tasks')
parser.add_argument('--learning_rate', type=float, default=0.00001)

parser.add_argument("--dataset_name", default="MSCOCO", type=str, help="MSCOCO/NUSWIDE")
parser.add_argument("--bit", default=16, type=int, help="16/32/64/128/256")
parser.add_argument("--prompt_mode", default='share', type=str, help="0: share, 1: separate")
parser.add_argument('--prompt_length', type=int, default=10, help='length of prompt.')
parser.add_argument('--prompt_add_length', type=int, default=3, help='length of prompt add.')
parser.add_argument('--old_dataset_code_is_useful', type=bool, default=True, help='Whether to use old dataset code.')

# loss
parser.add_argument('--quantify_loss', type=float, default=1.0, help='quantify loss weight')
parser.add_argument('--fine_loss', type=float, default=0.01, help='hash loss weight')
parser.add_argument('--coarse_loss', type=float, default=10.0, help='hash loss weight')
parser.add_argument('--task_distinct_loss', type=float, default=1.0, help='hash loss weight')
parser.add_argument('--exclude_loss', type=float, default=1.0, help='hash loss type')
parser.add_argument('--distill_loss', type=float, default=10.0, help='hash loss type')

args = parser.parse_args()

args.extend_learning_rate = args.learning_rate * 0.1

if 'MSCOCO' in args.dataset_name or 'MSCOCO_NoMean' in args.dataset_name:
    args.num_classes = 80
elif 'NUSWIDE' in args.dataset_name or 'NUSWIDE_NoMean' in args.dataset_name:
    args.num_classes = 81

def main():
    log = logger(args)
    set_seed(args.seed)
    args.dataset_path = os.path.join(args.data_path, args.dataset_name)
    # Create output directory for checkpoints and CSV results
    checkpoint_folder = os.path.join(args.output_dir, 'checkpoints', '{}'.format(args.runid))
    csv_folder = os.path.join(args.output_dir, 'csv_result', '{}'.format(args.runid))
    # Ensure the directories exist
    Path(checkpoint_folder).mkdir(parents=True, exist_ok=True)
    Path(csv_folder).mkdir(parents=True, exist_ok=True)

    checkpoints_path = checkpoint_folder + '/' + '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_hash_model.pth'.format(
        args.dataset_name, 
        args.bit, 
        args.prompt_mode, 
        args.learning_rate, 
        args.prompt_length,
        args.prompt_add_length, 
        args.quantify_loss, 
        args.fine_loss,
        args.coarse_loss,
        args.task_distinct_loss,
        args.exclude_loss, 
        args.distill_loss, 
        args.old_dataset_code_is_useful, 
        )
    args.image_old_dataset_hash_code_path = checkpoint_folder + '/' + '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_image_old_code.pt'.format(
        args.dataset_name, 
        args.bit, 
        args.prompt_mode, 
        args.learning_rate, 
        args.prompt_length,
        args.prompt_add_length, 
        args.quantify_loss, 
        args.fine_loss,
        args.coarse_loss,
        args.task_distinct_loss,
        args.exclude_loss, 
        args.distill_loss, 
        args.old_dataset_code_is_useful, 
        )
    args.text_old_dataset_hash_code_path = checkpoint_folder + '/' + '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_text_old_code.pt'.format(
        args.dataset_name, 
        args.bit, 
        args.prompt_mode, 
        args.learning_rate, 
        args.prompt_length,
        args.prompt_add_length, 
        args.quantify_loss, 
        args.fine_loss,
        args.coarse_loss,
        args.task_distinct_loss,
        args.exclude_loss, 
        args.distill_loss, 
        args.old_dataset_code_is_useful, 
        )
    args.label_old_dataset_hash_code_path = checkpoint_folder + '/' + '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_label_old_code.pt'.format(
        args.dataset_name, 
        args.bit, 
        args.prompt_mode, 
        args.learning_rate, 
        args.prompt_length,
        args.prompt_add_length, 
        args.quantify_loss, 
        args.fine_loss,
        args.coarse_loss,
        args.task_distinct_loss,
        args.exclude_loss, 
        args.distill_loss, 
        args.old_dataset_code_is_useful, 
        )
    result_dict = creat_result_dict(args)

    hashing_model = HashingModel(args).cuda()
    # Initialize loss function
    f_Fine_grained_similarity_loss = InfoNCELoss()

    # To store the hash codes of all previous tasks
    history_database_code_list_image = []
    history_database_code_list_text = []
    
    for task_index in range(args.num_tasks):
        input_data_par, dataloader = get_loader(args, task_index=task_index)

        train_model(args, log, 
                    hashing_model, 
                    input_data_par, dataloader, 
                    task_index, 
                    result_dict, checkpoints_path, 
                    f_Fine_grained_similarity_loss, 
                    history_database_code_list_image, 
                    history_database_code_list_text,
                    )
        log.info(f'The {task_index+1} task is trained')

    save_result_dict(args, result_dict, csv_folder)

if __name__ == '__main__':
    main()