import os
import math
import argparse
from collections import OrderedDict

import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizerFast

from utils.dataset import *
from utils.misc import *
from utils.data import *
from models.vae_gaussian import *
from models.vae_flow import *
from models.flow import add_spectral_norm, spectral_norm_power_iteration
from evaluation import *
from utils.objaverse_dataset import ObjaversePointCloudDataset
from utils.shapnet_own import ShapeNetCoreOwn

# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--model', type=str, default='flow', choices=['flow', 'gaussian'])
parser.add_argument('--model_size', type=str, default='original', choices=['original', 'big'])
parser.add_argument('--latent_dim', type=int, default=512)
parser.add_argument('--num_steps', type=int, default=100)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.02)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--truncate_std', type=float, default=2.0)
parser.add_argument('--latent_flow_depth', type=int, default=14)
parser.add_argument('--latent_flow_hidden_dim', type=int, default=256)
parser.add_argument('--num_samples', type=int, default=4)
parser.add_argument('--sample_num_points', type=int, default=2048)
parser.add_argument('--kl_weight', type=float, default=0.001)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--spectral_norm', type=eval, default=False, choices=[True, False])

# Datasets and loaders
parser.add_argument('--point_count', type=int, default=2048)
parser.add_argument('--dataset', type=str, default='objaverse', choices=['shapenet', 'objaverse'])
parser.add_argument('--dataset_path', type=str,
                    default='/Users/kostyalbalint/Documents/Egyetem/7.Felev/Szakdolgozat/pointClouds3000')
parser.add_argument('--dataset_file_ext', type=str, default='.npz')
# parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=64)
parser.add_argument('--annotations_file', type=str,
                    default='/Users/kostyalbalint/Documents/Egyetem/7.Felev/Szakdolgozat/objaverse_labeling/concatenated_annotations.npy')
parser.add_argument('--name_filter', type=str, default=None)
parser.add_argument('--load_to_mem', type=eval, default=False, choices=[True, False])

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=2e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=200 * THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=400 * THOUSAND)

# Training
parser.add_argument('--pretrained_ae', type=str, default=None)
parser.add_argument('--resume_ckpt', type=str, default=None)
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_gen')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--val_freq', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=30 * THOUSAND)
parser.add_argument('--test_size', type=int, default=50)
parser.add_argument('--tag', type=str, default=None)
args = parser.parse_args()
seed_all(args.seed)

# Resume training
if args.resume_ckpt:
    log_dir = args.resume_ckpt
    logger = get_logger('train', log_dir)
    logger.info(f'Resuming training for {log_dir}')
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
# Logging
elif args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='GEN_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, args)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()

is_resume = False
# Load previous args for resume
if args.resume_ckpt:
    is_resume = True
    ckpt = ckpt_mgr.load_latest()
    # args = ckpt['args']

logger.info(args)

# Datasets and loaders
logger.info('Loading datasets...')

# train_dset = ShapeNetCore(
#    path=args.dataset_path,
#    cates=args.categories,
#    split='train',
#    scale_mode=args.scale_mode,
# )

if args.dataset == 'objaverse':
    train_dset = ObjaversePointCloudDataset(annotations_file=args.annotations_file,
                                            file_ext=args.dataset_file_ext,
                                            pc_dir=args.dataset_path,
                                            split='train',
                                            scale_mode=args.scale_mode,
                                            name_filter=args.name_filter,
                                            load_to_mem=args.load_to_mem)

    val_dset = ObjaversePointCloudDataset(annotations_file=args.annotations_file,
                                          file_ext=args.dataset_file_ext,
                                          pc_dir=args.dataset_path,
                                          split='val',
                                          scale_mode=args.scale_mode,
                                          name_filter=args.name_filter,
                                          load_to_mem=args.load_to_mem)
elif args.dataset == 'shapenet':
    logger.info('Useing shapenet dataset')
    logger.info('Loading train set')
    train_dset = ShapeNetCoreOwn(
        path=args.dataset_path,
        annotations_path=args.annotations_file,
        split='train',
        scale_mode=args.scale_mode,
        transform=None,
        args = args,
        point_count=args.point_count
    )
    logger.info('Loading val set')
    val_dset = ShapeNetCoreOwn(
        path=args.dataset_path,
        annotations_path=args.annotations_file,
        split='val',
        scale_mode=args.scale_mode,
        transform=None,
        args = args,
        point_count=args.point_count
    )


args.latent_text_dim = train_dset.__getitem__(0)['latent_text'].shape[0]

logger.info('Dataset size: ' + str(train_dset.__len__()))

train_iter = get_data_iterator(DataLoader(
    train_dset,
    batch_size=args.train_batch_size,
    num_workers=0,
))

# Model
logger.info('Building model...')
if args.model == 'gaussian':
    model = GaussianVAE(args).to(args.device)
elif args.model == 'flow':
    model = FlowVAE(args).to(args.device)
logger.info(repr(model))
if args.spectral_norm:
    add_spectral_norm(model, logger=logger)

# Optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(),
                             lr=args.lr,
                             weight_decay=args.weight_decay
                             )
scheduler = get_linear_scheduler(
    optimizer,
    start_epoch=args.sched_start_epoch,
    end_epoch=args.sched_end_epoch,
    start_lr=args.lr,
    end_lr=args.end_lr
)


def map_AE_keys(state_dict):
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        if key.startswith('encoder'):
            # Remove 'encoder.' from the key and add it to the new_state_dict
            new_key = key[len('encoder.'):]
            new_state_dict[new_key] = value

    return new_state_dict


# Load previous state dicts
if is_resume:
    logger.info('Resumed training, loading back model state...')
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['others']['optimizer'])
    scheduler.load_state_dict(ckpt['others']['scheduler'])
elif args.pretrained_ae:
    logger.info('Loading back AE state dict...')
    ae_state = torch.load(args.pretrained_ae, map_location=args.device)
    state_dict = map_AE_keys(ae_state['state_dict'])
    model.encoder.load_state_dict(state_dict)


# Train, validate and test
def train(it):
    # Load data
    batch = next(train_iter)
    x = batch['pointcloud'].to(args.device)
    encoded_text = batch['latent_text'].to(args.device)

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()
    if args.spectral_norm:
        spectral_norm_power_iteration(model, n_power_iterations=1)

    # Forward
    kl_weight = args.kl_weight
    loss = model.get_loss(x, encoded_text, kl_weight=kl_weight, writer=writer, it=it)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f | KLWeight %.4f' % (
        it, loss.item(), orig_grad_norm, kl_weight
    ))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/kl_weight', kl_weight, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()


def tokenize_sentences(sentence):
    tokenizer = BertTokenizerFast.from_pretrained("setu4993/LEALLA-small")
    tokenizer_model = BertModel.from_pretrained("setu4993/LEALLA-small").to(args.device)
    tokenizer_model = tokenizer_model.eval()
    english_inputs = tokenizer([sentence], return_tensors="pt", padding=True, max_length=512, truncation=True).to(
        args.device)
    with torch.no_grad():
        english_outputs = tokenizer_model(**english_inputs).pooler_output

    return english_outputs.cpu().numpy()[0]


def validate_inspect(it):
    encoded_text = tokenize_sentences("Airplane plane flying jet")
    encoded_text = torch.tensor(np.resize(encoded_text, (args.num_samples, encoded_text.shape[0]))).to(args.device)

    z = torch.randn([args.num_samples, args.latent_dim]).to(args.device)

    x = model.sample(z, encoded_text, args.sample_num_points,
                     flexibility=args.flexibility)  # , truncate_std=args.truncate_std)
    writer.add_mesh('val/pointcloud', x, global_step=it, config_dict={
        'material': {
            'cls': 'PointsMaterial',
            'size': 0.015
        }
    })
    writer.flush()
    logger.info('[Inspect] Generating samples...')


def test(it):
    ref_pcs = []
    val_dset.shuffle()
    for i, data in enumerate(val_dset):
        if i >= args.test_size:
            break
        ref_pcs.append(data['pointcloud'].unsqueeze(0))
    ref_pcs = torch.cat(ref_pcs, dim=0)

    gen_pcs = []
    for i in tqdm(range(0, math.ceil(args.test_size / args.val_batch_size)), 'Generate'):
        with torch.no_grad():
            z = torch.randn([args.val_batch_size, args.latent_dim]).to(args.device)
            x = model.sample(z, args.sample_num_points, flexibility=args.flexibility)
            gen_pcs.append(x.detach().cpu())
    gen_pcs = torch.cat(gen_pcs, dim=0)[:args.test_size]

    # Denormalize point clouds, all shapes have zero mean.
    # [WARNING]: Do NOT denormalize!
    # ref_pcs *= val_dset.stats['std']
    # gen_pcs *= val_dset.stats['std']

    with torch.no_grad():
        results = compute_all_metrics(gen_pcs.to(args.device), ref_pcs.to(args.device), args.val_batch_size)
        results = {k: v.item() for k, v in results.items()}
        jsd = jsd_between_point_cloud_sets(gen_pcs.cpu().numpy(), ref_pcs.cpu().numpy())
        results['jsd'] = jsd

    # CD related metrics
    writer.add_scalar('test/Coverage_CD', results['lgan_cov-CD'], global_step=it)
    writer.add_scalar('test/MMD_CD', results['lgan_mmd-CD'], global_step=it)
    writer.add_scalar('test/1NN_CD', results['1-NN-CD-acc'], global_step=it)
    # EMD related metrics
    # writer.add_scalar('test/Coverage_EMD', results['lgan_cov-EMD'], global_step=it)
    # writer.add_scalar('test/MMD_EMD', results['lgan_mmd-EMD'], global_step=it)
    # writer.add_scalar('test/1NN_EMD', results['1-NN-EMD-acc'], global_step=it)
    # JSD
    writer.add_scalar('test/JSD', results['jsd'], global_step=it)

    # logger.info('[Test] Coverage  | CD %.6f | EMD %.6f' % (results['lgan_cov-CD'], results['lgan_cov-EMD']))
    # logger.info('[Test] MinMatDis | CD %.6f | EMD %.6f' % (results['lgan_mmd-CD'], results['lgan_mmd-EMD']))
    # logger.info('[Test] 1NN-Accur | CD %.6f | EMD %.6f' % (results['1-NN-CD-acc'], results['1-NN-EMD-acc']))
    logger.info('[Test] Coverage  | CD %.6f | EMD n/a' % (results['lgan_cov-CD'],))
    logger.info('[Test] MinMatDis | CD %.6f | EMD n/a' % (results['lgan_mmd-CD'],))
    logger.info('[Test] 1NN-Accur | CD %.6f | EMD n/a' % (results['1-NN-CD-acc'],))
    logger.info('[Test] JsnShnDis | %.6f ' % (results['jsd']))


# Main loop
logger.info('Start training...')
try:
    it = 1
    if is_resume:
        it = ckpt['others']['iteration'] + 1
    while it <= args.max_iters:
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            validate_inspect(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'iteration': it
            }
            ckpt_mgr.save(model, args, 0, others=opt_states, step=it)
        if it % args.test_freq == 0 or it == args.max_iters:
            # test(it)
            pass
        it += 1

except KeyboardInterrupt:
    logger.info('Terminating...')
