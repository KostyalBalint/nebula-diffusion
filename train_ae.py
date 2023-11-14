import os
import argparse
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from utils.dataset import *
from utils.misc import *
from utils.data import *
from utils.objaverse_dataset import ObjaversePointCloudDataset
from utils.transform import *
from models.autoencoder import *
from evaluation import EMD_CD

# Arguments
parser = argparse.ArgumentParser()
# Model arguments
parser.add_argument('--latent_dim', type=int, default=256)
parser.add_argument('--num_steps', type=int, default=200)
parser.add_argument('--beta_1', type=float, default=1e-4)
parser.add_argument('--beta_T', type=float, default=0.05)
parser.add_argument('--sched_mode', type=str, default='linear')
parser.add_argument('--flexibility', type=float, default=0.0)
parser.add_argument('--residual', type=eval, default=True, choices=[True, False])
parser.add_argument('--resume', type=str, default=None)

# Datasets and loaders
# Datasets and loaders
parser.add_argument('--dataset_path', type=str, default='/Users/kostyalbalint/Documents/Egyetem/7.Felev/Szakdolgozat/pointClouds3000')
parser.add_argument('--dataset_file_ext', type=str, default='.npz')
# parser.add_argument('--categories', type=str_list, default=['airplane'])
parser.add_argument('--scale_mode', type=str, default='shape_unit')
parser.add_argument('--train_batch_size', type=int, default=128)
parser.add_argument('--val_batch_size', type=int, default=32)
parser.add_argument('--rotate', type=eval, default=False, choices=[True, False])
parser.add_argument('--annotations_file', type=str,
                    default='/Users/kostyalbalint/Documents/Egyetem/7.Felev/Szakdolgozat/objaverse_labeling/concatenated_annotations.npy')
parser.add_argument('--name_filter', type=str, default=None)
parser.add_argument('--load_to_mem', type=eval, default=False, choices=[True, False])

# Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=10)
parser.add_argument('--end_lr', type=float, default=1e-4)
parser.add_argument('--sched_start_epoch', type=int, default=150 * THOUSAND)
parser.add_argument('--sched_end_epoch', type=int, default=300 * THOUSAND)

# Training
parser.add_argument('--initial_state_dict', type=str, default=None)
parser.add_argument('--resume_ckpt', type=str, default=None)
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs_ae')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=float('inf'))
parser.add_argument('--val_freq', type=float, default=1000)
parser.add_argument('--tag', type=str, default=None)
parser.add_argument('--num_val_batches', type=int, default=-1)
parser.add_argument('--num_inspect_batches', type=int, default=1)
parser.add_argument('--num_inspect_pointclouds', type=int, default=4)

# Validation
parser.add_argument('--val_set_percentage', type=float, default=0.1)
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
    log_dir = get_new_log_dir(args.log_root, prefix='AE_', postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()

is_resume = False
# Load previous args for resume
if args.resume_ckpt:
    is_resume = True
    ckpt = ckpt_mgr.load_latest()
    args = ckpt['args']

logger.info(args)

# Datasets and loaders
transform = None
if args.rotate:
    transform = RandomRotate(180, ['pointcloud'], axis=1)
logger.info('Transform: %s' % repr(transform))
logger.info('Loading datasets...')

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

args.latent_text_dim = train_dset.__getitem__(0)['latent_text'].shape[0]
logger.info('Dataset size: ' + str(train_dset.__len__()))

train_iter = get_data_iterator(DataLoader(
    train_dset,
    batch_size=args.train_batch_size,
    num_workers=0,
))
val_loader = DataLoader(val_dset, batch_size=args.val_batch_size, num_workers=0)

# Model
logger.info('Building model...')
#if args.resume is not None:
#    logger.info('Resuming from checkpoint...')
#    ckpt = torch.load(args.resume)
#    model = AutoEncoder(ckpt['args']).to(args.device)
#    model.load_state_dict(ckpt['state_dict'])
#else:
model = AutoEncoder(args).to(args.device)
logger.info(repr(model))

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

if args.initial_state_dict:
    logger.info(f'Loaded initial state dict from: {args.initial_state_dict}')
    model.load_state_dict(torch.load(args.initial_state_dict, map_location=args.device)['state_dict'])

# Load previous state dicts
if is_resume:
    model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['others']['optimizer'])
    scheduler.load_state_dict(ckpt['others']['scheduler'])

# Train, validate
def train(it):
    # Load data
    batch = next(train_iter)
    x = batch['pointcloud'].to(args.device)

    # Reset grad and model state
    optimizer.zero_grad()
    model.train()

    # Forward
    loss = model.get_loss(x)

    # Backward and optimize
    loss.backward()
    orig_grad_norm = clip_grad_norm_(model.parameters(), args.max_grad_norm)
    optimizer.step()
    scheduler.step()

    logger.info('[Train] Iter %04d | Loss %.6f | Grad %.4f ' % (it, loss.item(), orig_grad_norm))
    writer.add_scalar('train/loss', loss, it)
    writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], it)
    writer.add_scalar('train/grad_norm', orig_grad_norm, it)
    writer.flush()


def validate_loss(it):
    all_refs = []
    all_recons = []
    subset_percentage = args.val_set_percentage
    val_loader.dataset.shuffle()
    subset_size = int(len(val_loader) * subset_percentage)

    for i, batch in enumerate(tqdm(val_loader, desc='Validate', total=subset_size)):
        if i >= subset_size:
            tqdm._instances.clear()
            break
        if args.num_val_batches > 0 and i >= args.num_val_batches:
            break
        ref = batch['pointcloud'].to(args.device)
        shift = batch['shift'].to(args.device)
        scale = batch['scale'].to(args.device)
        with torch.no_grad():
            model.eval()
            code = model.encode(ref)
            recons = model.decode(code, ref.size(1), flexibility=args.flexibility)
        all_refs.append(ref * scale + shift)
        all_recons.append(recons * scale + shift)

    all_refs = torch.cat(all_refs, dim=0)
    all_recons = torch.cat(all_recons, dim=0)
    metrics = EMD_CD(all_recons, all_refs, batch_size=args.val_batch_size)
    cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()

    logger.info('[Val] Iter %04d | CD %.6f | EMD %.6f  ' % (it, cd, emd))
    writer.add_scalar('val/cd', cd, it)
    writer.add_scalar('val/emd', emd, it)
    writer.flush()

    return cd


def validate_inspect(it):
    sum_n = 0
    sum_chamfer = 0
    for i, batch in enumerate(tqdm(val_loader, desc='Inspect')):
        x = batch['pointcloud'].to(args.device)
        model.eval()
        code = model.encode(x)
        recons = model.decode(code, x.size(1), flexibility=args.flexibility).detach()

        sum_n += x.size(0)
        if i >= args.num_inspect_batches:
            break  # Inspect only 5 batch

    writer.add_mesh('original/pointcloud', batch['pointcloud'][:args.num_inspect_pointclouds], global_step=it, config_dict={
        'material': {
            'cls': 'PointsMaterial',
            'size': 0.05
        }
    })
    writer.add_mesh('val/pointcloud', recons[:args.num_inspect_pointclouds], global_step=it, config_dict={
        'material': {
            'cls': 'PointsMaterial',
            'size': 0.05
        }
    })
    writer.flush()


# Main loop
logger.info('Start training...')
try:
    it = 1
    if is_resume:
        it = ckpt['others']['iteration']
    while it <= args.max_iters:
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            with torch.no_grad():
                cd_loss = validate_loss(it)
                validate_inspect(it)
            opt_states = {
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'iteration': it
            }
            ckpt_mgr.save(model, args, cd_loss, opt_states, step=it)
        it += 1

except KeyboardInterrupt:
    logger.info('Terminating...')
