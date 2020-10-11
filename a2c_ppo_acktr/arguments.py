import jsonargparse as argparse

import torch


class NoneableType(object):
    def __init__(self, type):
        self._type = type

    def __call__(self, value):
        if value is not None:
            return self._type(value)
        return value


def get_args(sysargs):
    parser = argparse.ArgumentParser(description='RL')

    parser.add_argument('--config',
                        help="Read configurations from file (will override all configurations which are passed "
                             "before this option)", action=argparse.ActionConfigFile)

    # Globals
    parser.add_argument('--globals.seed', type=int, default=1,
                        help='Random seed (default: 1)')
    parser.add_argument('--globals.use-cuda', action='store_true', default=True,
                        help='Enables CUDA training')
    parser.add_argument('--globals.cuda-deterministic', action='store_true', default=True,
                        help="Sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument('--globals.num-processes', type=int, default=16,
                        help='How many training CPU processes to use (default: 16)')
    parser.add_argument('--globals.num-steps', type=int, default=5,
                        help='Number of forward steps in A2C (default: 5)')
    parser.add_argument('--globals.num-mini-batch', type=int, default=32,
                        help='Number of batches for ppo (default: 32)')

    # env
    parser.add_argument('--env.name', default='PongNoFrameskip-v4',
                        help='Environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument('--env.params-file', type=NoneableType(str), default=None,
                        help='Load environment sampler parameters from file')
    parser.add_argument('--env.data-folder-path', default=None,
                        help='Set the data folder path if the env uses external data (e.g. curriculum env)')
    parser.add_argument('--env.num-env-steps', type=int, default=10e6,
                        help='Number of environment steps to train (default: 10e6)')
    parser.add_argument('--env.normalize-obs', action='store_true', default=True,
                        help='Normalize state observations')
    parser.add_argument('--env.add-timestep', action='store_true', default=False,
                        help='add timestep to observations')
    parser.add_argument('--env.num-framestack', type=int, default=1,
                        help='Num frame stack')

    # learning.rl
    parser.add_argument('--learning.rl.algo', default='ppo',
                        help='Algorithm to use: a2c | ppo | acktr')
    parser.add_argument('--learning.rl.gamma', type=float, default=0.99,
                        help='Discount factor for rewards (default: 0.99)')
    parser.add_argument('--learning.rl.entropy-coef', type=float, default=0.01,
                        help='Entropy term coefficient (default: 0.01)')
    parser.add_argument('--learning.rl.value-loss-coef', type=float, default=0.5,
                        help='Value loss coefficient (default: 0.5)')
    parser.add_argument('--learning.rl.max-grad-norm', type=float, default=0.5,
                        help='Max norm of gradients (default: 0.5)')

    # learning.rl.ppo
    parser.add_argument('--learning.rl.ppo.epoch', type=int, default=4,
                        help='Number of ppo epochs (default: 4)')
    parser.add_argument('--learning.rl.ppo.use-linear-clip-decay', action='store_true', default=False,
                        help='Use a linear schedule on the ppo clipping parameter')
    parser.add_argument('--learning.rl.ppo.use-linear-clip-decay-less', action='store_true', default=False,
                        help='Use a linear schedule on the ppo clipping parameter')
    parser.add_argument('--learning.rl.ppo.clip-param', type=float, default=0.2,
                        help='PPO clip parameter (default: 0.2)')

    # learning.rl.actor_critic
    parser.add_argument('--learning.rl.actor-critic.snapshot', type=NoneableType(str), default=None,
                        help='snapshot for pretrained policy (default: None)')
    parser.add_argument('--learning.rl.actor-critic.combi-policy', action='store_true', default=True,
                        help='Use image and state input together')
    parser.add_argument('--learning.rl.actor-critic.recurrent-policy', action='store_true', default=False,
                        help='use a recurrent policy')
    parser.add_argument('--learning.rl.actor-critic.network-architecture', default='symm',
                        help='train policy with image and vf with full state')
    parser.add_argument('--learning.rl.actor-critic.cnn-architecture', default='nature',
                        help='which cnn architecture')

    # learning.rl.gae
    parser.add_argument('--learning.rl.gae.enable', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--learning.rl.gae.tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')

    # learning.optimizer
    parser.add_argument('--learning.optimizer.lr', type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument('--learning.optimizer.eps', type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--learning.optimizer.alpha', type=float, default=0.99,
                        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument('--learning.optimizer.use-linear-lr-decay', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--learning.optimizer.use-linear-lr-decay-half', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--learning.optimizer.use-linear-lr-decay-less', action='store_true', default=False,
                        help='use a linear schedule on the learning rate')
    parser.add_argument('--learning.optimizer.use-sr-schedule', action='store_true', default=False,
                        help='use success rate to decrease learning rate')

    # learning.curriculum
    parser.add_argument('--learning.curriculum.enable', action='store_true', default=True,
                        help='use curriculum')
    parser.add_argument('--learning.curriculum.adaptive-curriculum', action='store_true', default=False,
                        help='use an adaptive curriculum')
    parser.add_argument('--learning.curriculum.desired-rew-region-lo', type=float, default=0.4,
                        help='the desired range of train rewards')
    parser.add_argument('--learning.curriculum.desired-rew-region-hi', type=float, default=0.6,
                        help='the desired range of train rewards')
    parser.add_argument('--learning.curriculum.incr', type=float, default=0.002,
                        help='the increment (decrement) if train rewards go out of desired region')
    parser.add_argument('--learning.curriculum.rew-q-len', type=int, default=20,
                        help='length of reward queue')

    # learning.consistency_loss
    parser.add_argument('--learning.consistency_loss.enable', default=False, action="store_true",
                        help="If set, the consistency loss is used during training")
    parser.add_argument('--learning.consistency_loss.augmenter', type=str, default=None,
                        help='Specifies which transformer is to be used for augmenting the data for'
                             'calculating the augmentation loss during training. '
                             'Default: color_transformer')
    parser.add_argument('--learning.consistency_loss.transformer.args_path', type=str, default=None,
                        help='Path to .yaml file containing transformer arguments. If not set, some default '
                             'values will be applied.')
    parser.add_argument('--learning.consistency_loss.dataset-folder', default=None,
                        help="Root folder containing dataset data")
    parser.add_argument('--learning.consistency_loss.dataloader-batch-size', default='same',
                        help="Batch size of dataset dataloader. If set to same, dataloader will"
                             "use the same batch size as the reinforcement learning policy.")
    parser.add_argument('--learning.consistency_loss.loss-weight', default=1.0, type=float,
                        help="Set a constant factor for weighting the augmentation loss. Range [0.0, 1.0]."
                             "if set, it will override augmentation loss function (and params)")
    parser.add_argument('--learning.consistency_loss.loss-weight-function-params', default=None,
                        help=".npz or .npy file path containing polynomial parameters (compatible with numpy.poly1d)."
                             "If set, this will override augmentation-loss-weight argument")
    parser.add_argument('--learning.consistency_loss.use-cnn-loss', default=False, action="store_true",
                        help="If set, the augmentation loss is calculated over the CNN output instead of output action")
    parser.add_argument('--learning.consistency_loss.clip-aug-actions', default=False, action="store_true",
                        help="If set, the action calculated in the augmentation loss will be clipped to "
                             "the action range used in the environment")
    parser.add_argument('--learning.consistency_loss.use-action-loss-as-weight', default=False, action="store_true",
                        help="If set, the consistency loss weight will be derived from the action loss "
                             "by a moving average. Can be combined with loss_weight and /or loss_weight_function_params.")
    parser.add_argument('--learning.consistency_loss.eval_target_env', default=None,
                        help="If set, an additional evaluation will be performed on the given environment with the set eval_interval")
    parser.add_argument('--learning.consistency_loss.force-disable-consistency', default=False, action="store_true",
                        help="If set to to true, the action loss is calculated without consistency loss")
    parser.add_argument('--learning.consistency_loss.target-model-update-frequency', default=-1,
                        help="If set to value > 0, the target model in the consistency loss will be updated with given "
                             "frequency")
    parser.add_argument('--learning.consistency_loss.target-model-discount', default=1.0,
                        help="Discount of 1.0: use new weights for target model. Discount < 1.0: exponential moving avg.")

    # experiment
    parser.add_argument('--experiment.num-bc-epochs', type=int, default=1000,
                        help='Snapshot for pretrained policy (default: None)')
    parser.add_argument('--experiment.save-interval', type=int, default=100,
                        help='save interval, one save per n updates (default: 100)')
    parser.add_argument('--experiment.log-interval', type=int, default=10,
                        help='log interval, one log per n updates (default: 10)')
    parser.add_argument('--experiment.eval-interval', type=NoneableType(int), default=None,
                        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument('--experiment.vis', action='store_true', default=False,
                        help='enable visdom visualization')
    parser.add_argument('--experiment.vis-interval', type=int, default=1,
                        help='vis interval, one log per n updates (default: 100)')
    parser.add_argument('--experiment.vis_port', type=int, default=8097,
                        help='port to run the visualization server on (default: 8097)')
    parser.add_argument('--experiment.save-eval-images', action='store_true', default=False,
                        help='Save evaluation images')
    parser.add_argument('--experiment.save-train-images', action='store_true', default=False,
                        help='Save training images every save_interval steps')
    parser.add_argument('--experiment.consistency_loss.use-augmentation-loss', action='store_true', default=False,
                        help='Use data augmentation with separate loss during training')
    parser.add_argument('--experiment.root-dir', default='/tmp/a2c_ppo_acktr',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--experiment.save-dir', default='/tmp/a2c_ppo_acktr',
                        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument('--experiment.log-dir', default='/tmp/a2c_ppo_acktr/',
                        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument('--experiment.tag', default=None, help="Tag gets appended to training name")

    args = parser.parse_args(sysargs)
    args.globals.cuda_enabled = args.globals.use_cuda and torch.cuda.is_available()
    return args
