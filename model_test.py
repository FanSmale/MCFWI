# -*- coding: utf-8 -*-

from func.datasets_reader import *
from model_train import determine_network, get_dataset_config, pain_openfwi_velocity_model, pain_seg_velocity_model


def parse_args():
    parser = argparse.ArgumentParser(description="testing script for seismic wave velocity inversion network")

    parser.add_argument('--model_type', type=str, default='MCFWI',
                        help='net: DDNet, InversionNet, MCFWI')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the .pth weight file for testing')

    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (placeholder for optimizer initialization during testing)')

    parser.add_argument('--dataset_name', type=str, default='CurveVelA', help='datasets')
    parser.add_argument('--data_dir', type=str, default='H:/OpenFWI_data/CurveVelA/', help='Path to the test data')
    parser.add_argument('--test_size', type=int, default=1000, help='Total number of test samples')
    parser.add_argument('--batch_size', type=int, default=2, help='Test batch size (samples per GPU batch)')

    parser.add_argument('--mode', type=str, choices=['batch', 'single'], default='batch',
                        help='Run mode: batch (evaluation) or single (visualization)')
    parser.add_argument('--select_id', type=int, default=66, help='Specified sample ID in single mode')
    parser.add_argument('--out_dir', type=str, default='./results/', help='Save path for result images in single mode')

    return parser.parse_args()


def load_test_dataset(args, config):
    print("---------------------------------")
    print("· Loading test data...")


    num_files_to_read = max(1, args.test_size // 500)
    data_set, label_sets = batch_read_npyfile(args.data_dir, 1, num_files_to_read, "test")


    data_set, label_sets = data_set[:args.test_size], label_sets[:args.test_size]

    for i in range(data_set.shape[0]):
        vm = label_sets[i][0]
        label_sets[i][0] = (vm - np.min(vm)) / (np.max(vm) - np.min(vm))

    print(f"· Total test samples: {args.test_size}.")
    print("---------------------------------")

    seis_and_vm = data_utils.TensorDataset(torch.from_numpy(data_set).float(), torch.from_numpy(label_sets).float())
    loader = data_utils.DataLoader(seis_and_vm, batch_size=args.batch_size, shuffle=False)

    return loader, data_set, label_sets


def batch_test(args, config):
    loader, _, _ = load_test_dataset(args, config)

    print(f"-> Loading test model: {args.model_path}")
    model_net, device, _, _, _ = determine_network(args, config)
    model_net.eval()

    mse_record = np.zeros(args.test_size)
    mae_record = np.zeros(args.test_size)
    uqi_record = np.zeros(args.test_size)
    lpips_record = np.zeros(args.test_size)
    ssim_record = np.zeros(args.test_size)

    counter = 0
    lpips_object = lpips.LPIPS(net='alex', version="0.1")

    cur_node_time = time.time()
    with torch.no_grad():
        for i, (seis_image, gt_vmodel) in enumerate(loader):
            if torch.cuda.is_available():
                seis_image, gt_vmodel = seis_image.cuda(non_blocking=True), gt_vmodel.cuda(non_blocking=True)

            if args.model_type in ["DDNet", "DDNet70"]:
                outputs, _ = model_net(seis_image, config['model_dim'])
            elif args.model_type == "InversionNet":
                outputs = model_net(seis_image)
            else:
                outputs = model_net(seis_image, config['model_dim'])

            pd_vmodel = outputs.cpu().numpy()
            pd_vmodel = np.where(pd_vmodel > 0.0, pd_vmodel, 0.0)
            gt_vmodel_np = gt_vmodel.cpu().numpy()

            current_batch_size = seis_image.size(0)
            for k in range(current_batch_size):
                if counter >= args.test_size:
                    break

                pd_vmodel_of_k = pd_vmodel[k, 0, :, :]
                gt_vmodel_of_k = gt_vmodel_np[k, 0, :, :]

                mse_record[counter] = run_mse(pd_vmodel_of_k, gt_vmodel_of_k)
                mae_record[counter] = run_mae(pd_vmodel_of_k, gt_vmodel_of_k)
                uqi_record[counter] = run_uqi(gt_vmodel_of_k, pd_vmodel_of_k)
                lpips_record[counter] = run_lpips(gt_vmodel_of_k, pd_vmodel_of_k, lpips_object)
                ssim_record[counter] = run_ssim(gt_vmodel_of_k, pd_vmodel_of_k)

                print(
                    f'sample {counter} -> MSE: {mse_record[counter]:.6f} | MAE: {mae_record[counter]:.6f} | SSIM: {ssim_record[counter]:.6f}')
                counter += 1

    time_elapsed = time.time() - cur_node_time
    print("========= Batch Test Summary =========")
    print(f"Average MSE:   {mse_record.mean():.6f}")
    print(f"Average MAE:   {mae_record.mean():.6f}")
    print(f"Average UQI:   {uqi_record.mean():.6f}")
    print(f"Average LPIPS: {lpips_record.mean():.6f}")
    print(f"Average SSIM:  {ssim_record.mean():.7f}")
    print(f"Total test time: {time_elapsed:.2f}s ({time_elapsed / args.test_size:.4f}s/sample)")


def single_test(args, config):
    print(f"-> Loading model for single test: {args.model_path}")
    os.makedirs(args.out_dir, exist_ok=True)
    model_net, device, _, _, _ = determine_network(args, config)
    model_net.eval()

    seismic_data, velocity_model, _ = single_read_npyfile(args.data_dir, [1, args.select_id], train_or_test="test")
    max_velocity, min_velocity = np.max(velocity_model), np.min(velocity_model)
    velocity_model = (velocity_model - np.min(velocity_model)) / (np.max(velocity_model) - np.min(velocity_model))

    lpips_object = lpips.LPIPS(net='alex', version="0.1")

    seismic_data_tensor = torch.from_numpy(np.array([seismic_data])).float()
    if torch.cuda.is_available():
        seismic_data_tensor = seismic_data_tensor.cuda(non_blocking=True)

    with torch.no_grad():
        if args.model_type in ["DDNet", "DDNet70"]:
            predicted_vmod_tensor, _ = model_net(seismic_data_tensor, config['model_dim'])
        elif args.model_type == "InversionNet":
            predicted_vmod_tensor = model_net(seismic_data_tensor)
        else:
            predicted_vmod_tensor = model_net(seismic_data_tensor, config['model_dim'])

    predicted_vmod = predicted_vmod_tensor.cpu().numpy()[0][0]
    predicted_vmod = np.where(predicted_vmod > 0.0, predicted_vmod, 0.0)

    mse = run_mse(predicted_vmod, velocity_model)
    mae = run_mae(predicted_vmod, velocity_model)
    uqi = run_uqi(velocity_model, predicted_vmod)
    lpi = run_lpips(velocity_model, predicted_vmod, lpips_object)
    ssim = run_ssim(velocity_model, predicted_vmod)

    print(f"========= Test Results for Sample {args.select_id} =========")
    print(f"MSE: {mse:.6f} | MAE: {mae:.6f} | UQI: {uqi:.6f} | LPIPS: {lpi:.6f} | SSIM: {ssim:.6f}")

    save_name = f"pred_id{args.select_id}.png"
    gt_save_name = f"gt_id{args.select_id}.png"

    if args.dataset_name in ['SEGSalt', 'SEGSimulation']:
        pain_seg_velocity_model(velocity_model, min_velocity, max_velocity, args.out_dir, gt_save_name)
        pain_seg_velocity_model(predicted_vmod, min_velocity, max_velocity, args.out_dir, save_name)
    else:
        minV = np.min(min_velocity + velocity_model * (max_velocity - min_velocity))
        maxV = np.max(min_velocity + velocity_model * (max_velocity - min_velocity))
        pain_openfwi_velocity_model(min_velocity + velocity_model * (max_velocity - min_velocity), minV, maxV,
                                    args.out_dir, gt_save_name)
        pain_openfwi_velocity_model(min_velocity + predicted_vmod * (max_velocity - min_velocity), minV, maxV,
                                    args.out_dir, save_name)

def main():
    args = parse_args()
    args.resume = args.model_path
    config = get_dataset_config(args.dataset_name)

    if args.mode == 'batch':
        batch_test(args, config)
    elif args.mode == 'single':
        single_test(args, config)


if __name__ == "__main__":
    main()

#python model_test.py --mode batch --model_path "Replace it with your own path" --test_size 1000
#python model_test.py --mode single --select_id 50 --model_path "Replace it with your own path"
