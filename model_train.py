# -*- coding: utf-8 -*-

import json
import datetime
from torch.utils.tensorboard import SummaryWriter
from func.datasets_reader import *
from func.comparison_net import InversionNet
from func.net import DDNet70Model
from loss import LossSwin
from MCFWI import MCFWI


def parse_args():
    parser = argparse.ArgumentParser(description="training script for seismic wave velocity inversion network")
    parser.add_argument('--model_type', type=str, default='MCFWI',
                        help='net: DDNet70, InversionNet, MCFWI')
    parser.add_argument('--epochs', type=int, default=150, help='epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch_size')
    parser.add_argument('--lr', type=float, default=0.001, help='lr')
    parser.add_argument('--val_ratio', type=float, default=0, help='Validation set split ratio')
    parser.add_argument('--patience', type=int, default=5, help='Early Stopping Patience')
    parser.add_argument('--dataset_name', type=str, default='CurveVelA',
                        help='datasets (e.g. CurveVelA)')
    parser.add_argument('--data_dir', type=str, default='H:/OpenFWI_data/CurveVelA/', help='Path to training data')
    parser.add_argument('--out_dir', type=str, default='datasets/CurveVelA_new/', help='Base output directory for models and logs')

    parser.add_argument('--train_size', type=int, default=24000,
                        help='Total number of samples to load. For example, if each .npy file contains 500 samples, passing 5000 will load 10 .npy files.')

    parser.add_argument('--save_freq', type=int, default=10, help='Save regular model every N epochs')
    parser.add_argument('--resume', type=str, default='', help='Path to the model checkpoint (.pth) for resuming training')
    return parser.parse_args()


def get_dataset_config(dataset_name):
    config = {'classes': 1}
    if dataset_name in ['SEGSimulation', 'SEGSalt']:
        config['data_dim'] = [400, 301]
        config['model_dim'] = [201, 301]
        config['inchannels'] = 29
    else:
        config['data_dim'] = [1000, 70]
        config['model_dim'] = [70, 70]
        config['inchannels'] = 5
    return config


def determine_network(args, config):
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    gpus = [0]
    inchannels, classes = config['inchannels'], config['classes']

    if args.model_type == "DDNet70":
        net_model = DDNet70Model(n_classes=classes, in_channels=inchannels, is_deconv=True, is_batchnorm=True)
    elif args.model_type == "InversionNet":
        net_model = InversionNet()
    elif args.model_type == "MCFWI":
        net_model = MCFWI()
    else:
        raise ValueError(f"不支持的 model_type: {args.model_type}")

    if torch.cuda.is_available():
        net_model = torch.nn.DataParallel(net_model, device_ids=gpus).cuda()

    opt_lr = 0.0001 if args.model_type == "InversionNet" else args.lr
    optimizer = torch.optim.Adam(net_model.parameters(), lr=opt_lr)

    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            net_model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        else:
            net_model.load_state_dict(checkpoint)

    return net_model, device, optimizer, start_epoch, best_val_loss


def load_and_split_dataset(args, config):
    print("---------------------------------")
    print(f"· Loading {args.dataset_name} dataset and splitting into train and validation sets...")

    num_files_to_read = math.ceil(args.train_size / 500)
    print(f"· Calculated that {num_files_to_read} .npy files need to be loaded based on train_size = {args.train_size}")
    data_set, label_sets = batch_read_npyfile(args.data_dir, 1, num_files_to_read, "train")

    data_set = data_set[:args.train_size]
    label_sets = label_sets[:args.train_size]

    print("· Normalizing velocity model...")
    for i in range(data_set.shape[0]):
        vm = label_sets[i][0]
        label_sets[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

    full_dataset = data_utils.TensorDataset(torch.from_numpy(data_set).float(), torch.from_numpy(label_sets).float())
    total_size = len(full_dataset)
    val_size = int(total_size * args.val_ratio)
    train_size_split = total_size - val_size

    train_dataset, val_dataset = data_utils.random_split(full_dataset, [train_size_split, val_size])

    train_loader = data_utils.DataLoader(train_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=True)

    val_loader = data_utils.DataLoader(val_dataset, batch_size=args.batch_size, pin_memory=True, shuffle=False)

    print(f"· Final total samples used: {total_size} (Train: {train_size_split}, Val: {val_size})")
    print("---------------------------------")
    return train_loader, val_loader


def main():
    args = parse_args()
    ds_config = get_dataset_config(args.dataset_name)
    model_dim = ds_config['model_dim']

    now = datetime.datetime.now()
    date_str, time_str = now.strftime("%Y-%m-%d"), now.strftime("%H-%M-%S")
    save_base_dir = os.path.join(args.out_dir, "models", date_str, time_str)
    ckpt_dir, log_dir = os.path.join(save_base_dir, "checkpoints"), os.path.join(save_base_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(save_base_dir, 'hparams.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

    writer = SummaryWriter(log_dir=log_dir)

    model, device, optimizer, start_epoch, best_val_loss = determine_network(args, ds_config)
    train_loader, val_loader = load_and_split_dataset(args, ds_config)
    criterion = LossSwin(weights=[1, 1e6] if args.model_type == "DDNet" else [1, 1], entropy_weight=[1, 1])

    early_stop_counter, prev_val_loss = 0, None
    min_delta = args.lr / 10.0

    print(f"======== 开始训练 {args.model_type} ========")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss, cur_node_time = 0.0, time.time()

        for i, (images, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
            optimizer.zero_grad()

            if args.model_type in ["MCFWI", "DDNet70", "InversionNet"]:
                outputs = model(images, model_dim)
            else:
                outputs = model(images)

            if isinstance(outputs, (list, tuple)):
                out1 = outputs[0]
                out2 = outputs[1] if len(outputs) > 1 else outputs[0]
            else:
                out1 = outputs
                out2 = outputs

            loss = criterion(out1, out2, labels, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()

        if len(val_loader) > 0:
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    if torch.cuda.is_available():
                        images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)

                    if args.model_type in ["MCFWI", "DDNet70", "InversionNet"]:
                        outputs = model(images, model_dim)
                    else:
                        outputs = model(images)

                    if isinstance(outputs, (list, tuple)):
                        out1 = outputs[0]
                        out2 = outputs[1] if len(outputs) > 1 else outputs[0]
                    else:
                        out1 = outputs
                        out2 = outputs

                    val_loss += criterion(out1, out2, labels, labels).item()
            avg_val_loss = val_loss / len(val_loader)
        else:
            avg_val_loss = avg_train_loss

        writer.add_scalars('Loss', {'Train': avg_train_loss, 'Validation': avg_val_loss}, epoch + 1)

        time_elapsed = time.time() - cur_node_time
        print(
            f'Epoch [{epoch + 1}/{args.epochs}] Time: {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}')

        state_dict = {
            'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': avg_val_loss, 'best_val_loss': best_val_loss
        }

        if avg_val_loss < best_val_loss:
            best_val_loss, state_dict['best_val_loss'] = avg_val_loss, avg_val_loss
            torch.save(state_dict, os.path.join(ckpt_dir, 'best_model.pth'))
            with open(os.path.join(save_base_dir, 'best_record.txt'), 'w', encoding='utf-8') as f:
                f.write(f"Best Epoch: {epoch + 1}\n")
                f.write(f"Best Val Loss: {best_val_loss:.6f}\n")
            print(f" -> Best model updated (Loss: {best_val_loss:.5f})")

        torch.save(state_dict, os.path.join(ckpt_dir, 'latest_model.pth'))
        if (epoch + 1) % args.save_freq == 0:
            torch.save(state_dict, os.path.join(ckpt_dir, f'model_epoch_{epoch + 1}.pth'))

        if prev_val_loss is not None:
            if abs(prev_val_loss - avg_val_loss) < min_delta:
                early_stop_counter += 1
                print(f" -> Early stopping count: {early_stop_counter}/{args.patience}")
            else:
                early_stop_counter = 0
        prev_val_loss = avg_val_loss

        if early_stop_counter >= args.patience:
            print("======== Early stopping triggered. Stopping training! ========")
            break

    writer.close()


if __name__ == "__main__":
    main()


#python model_train.py --batch_size 16 --epochs 2 --train_size 500 --model_type MCFWI
#tensorboard --logdir=Replace it with your own path