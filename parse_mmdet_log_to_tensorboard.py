import json
from torch.utils.tensorboard import SummaryWriter
import os
import argparse

def parse_json_to_tensorboard(json_log_path, tb_log_dir):
    """
    Parses an MMDetection JSON log file and writes data to TensorBoard.
    """
    if not os.path.exists(json_log_path):
        print(f"Error: JSON log file not found at {json_log_path}")
        return

    # Ensure the TensorBoard log directory exists
    os.makedirs(tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f"TensorBoard logs will be saved to: {tb_log_dir}")

    # Determine if iters_per_epoch is needed (for older MMDetection logs if 'iter' is per-epoch)
    # Modern MMDetection json logs often have a global iteration/step count.
    # We will assume 'iter' or 'step' from the log is the primary x-axis for training losses.
    # And 'epoch' is the x-axis for validation metrics.

    # Store unique epochs for validation loss averaging if needed (MMDet usually logs val metrics per epoch)
    # val_losses_by_epoch = {}

    with open(json_log_path, 'r') as f:
        for line_idx, line in enumerate(f):
            try:
                log_entry = json.loads(line.strip())
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON line {line_idx+1}: {line.strip()}")
                continue

            mode = log_entry.get('mode', 'train') # Default to train if mode is not specified
            epoch = log_entry.get('epoch')
            
            # MMDetection 3.x uses 'step', MMDetection 2.x uses 'iter'
            # 'iter' in 2.x can be iter_per_epoch or global_iter depending on runner
            # We prioritize 'step' if available, else 'iter'.
            current_iter_or_step = log_entry.get('step', log_entry.get('iter'))

            if mode == 'train' and current_iter_or_step is not None:
                # Training losses (logged per iteration)
                for key, value in log_entry.items():
                    if 'loss' in key: # Catches 'loss', 'loss_cls', 'loss_bbox', 'loss_mask', etc.
                        writer.add_scalar(f'train/{key}', float(value), current_iter_or_step)
                    elif key == 'lr':
                        # LR can be a single value or a dict for different param groups
                        if isinstance(value, dict):
                            # Log each learning rate if multiple, or just the first common one
                            for k_lr, v_lr in value.items():
                                writer.add_scalar(f'train/lr_{k_lr}', float(v_lr), current_iter_or_step)
                                break # Often, just logging the first one is enough for a general trend
                        elif isinstance(value, list): # MMDetection 2.x
                             writer.add_scalar(f'train/{key}', float(value[0]), current_iter_or_step)
                        else:
                            writer.add_scalar(f'train/{key}', float(value), current_iter_or_step)
            
            elif mode == 'val' and epoch is not None:
                # Validation metrics (logged per epoch)
                # For VOC mAP, common keys: 'pascal_voc/mAP', 'pascal_voc/AP50'
                # For COCO mAP: 'coco/bbox_mAP', 'coco/bbox_mAP_50'
                # For validation loss (if logged, MMDetection doesn't always log overall val loss to json by default)
                
                # General metric handling
                for key, value in log_entry.items():
                    if 'mAP' in key or 'AP50' in key or 'AR@' in key: # Common metric patterns
                        # Sanitize key for TensorBoard tag
                        tb_key = key.replace('/', '_') 
                        writer.add_scalar(f'val/{tb_key}', float(value), epoch)
                    elif key == 'loss' and 'val' in log_entry.get('data_prefix', ''): # Less common, but if val loss is logged
                        writer.add_scalar('val/loss', float(value), epoch)
                    elif key.startswith('loss_') and 'val' in log_entry.get('data_prefix', ''): # e.g. val_loss_cls
                        writer.add_scalar(f'val/{key}', float(value), epoch)
                        
                # Specifically for VOC if not caught by generic above
                if 'pascal_voc/mAP' in log_entry:
                    writer.add_scalar('val/pascal_voc_mAP', float(log_entry['pascal_voc/mAP']), epoch)
                if 'pascal_voc/AP50' in log_entry:
                    writer.add_scalar('val/pascal_voc_AP50', float(log_entry['pascal_voc/AP50']), epoch)

    writer.close()
    print("Finished parsing log and writing TensorBoard events.")
    print(f"To view, run: tensorboard --logdir {os.path.dirname(tb_log_dir)} (or the exact tb_log_dir itself)")
    print(f"Example: tensorboard --logdir {tb_log_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Parse MMDetection JSON log to TensorBoard.")
    parser.add_argument('json_log_path', type=str, help="Path to the MMDetection JSON log file.")
    parser.add_argument('--tb_log_dir', type=str, default=None,
                        help="Directory to save TensorBoard event files. "
                             "Defaults to a 'tensorboard_logs' subdirectory next to the JSON file.")
    args = parser.parse_args()

    if args.tb_log_dir is None:
        # Default to a subdir in the same directory as the json log
        base_dir = os.path.dirname(args.json_log_path)
        tb_dir = os.path.join(base_dir, 'tensorboard_manual_logs')
    else:
        tb_dir = args.tb_log_dir
    
    parse_json_to_tensorboard(args.json_log_path, tb_dir)
