"""
This mode performs a test, where a trained model is evaluated on an environment with
increasing difficulty (increasing domain randomization, augmentation).
"""

import datetime
import json
import os
import random
import sys

import numpy as np
from a2c_ppo_acktr.augmentation.randaugment import RANDAUGMENT_MAP
from tensorboardX import SummaryWriter
from tqdm import tqdm

import a2c_ppo_acktr.augmentation.augmenters as augmenters
import a2c_ppo_acktr.augmentation.randaugment as randaugment
from a2c_ppo_acktr.augmentation.wrappers import AugmentationObservationWrapper
from a2c_ppo_acktr.play_model import Model, build_env, render_obs
from gym_grasping.envs import CurriculumEnvLog
from gym_grasping.envs.utils import import_env_params_from_file
from gym_grasping.scripts.utils.args import get_and_create_output_dir, get_base_argument_parser


def set_env_domain_randomization(grasping_env, domain_rand_amount):
    # Visual Domain Randomization
    grasping_env.env_params.set_variable_difficulty_r("vis/block_red", domain_rand_amount)
    grasping_env.env_params.set_variable_difficulty_r("vis/block_blue", domain_rand_amount)
    grasping_env.env_params.set_variable_difficulty_r("vis/table_green", domain_rand_amount)
    grasping_env.env_params.set_variable_difficulty_r("vis/brightness", domain_rand_amount)
    grasping_env.env_params.set_variable_difficulty_r("vis/contrast", domain_rand_amount)
    grasping_env.env_params.set_variable_difficulty_r("vis/color", domain_rand_amount)
    grasping_env.env_params.set_variable_difficulty_r("vis/shaprness", domain_rand_amount)
    grasping_env.env_params.set_variable_difficulty_r("vis/blur", domain_rand_amount)
    grasping_env.env_params.set_variable_difficulty_r("vis/hue", domain_rand_amount)
    grasping_env.env_params.set_variable_difficulty_r("vis/light_direction", domain_rand_amount)

    # Set task difficulty to maximum
    grasping_env.env_params.set_variable_difficulty_mu("geom/object_to_gripper", 1)
    grasping_env.env_params.set_variable_difficulty_r("geom/object_to_gripper", 1)


def setup_randaugment_augmentation_list(randaugment_augs):
    if randaugment_augs is None:
        return randaugment.AUGMENTATION_LIST_SMALL_RANGE
    else:
        augs = []
        for rand_aug in randaugment_augs:
            augs.append(RANDAUGMENT_MAP[rand_aug])
        return augs


if __name__ == '__main__':
    parser = get_base_argument_parser()
    parser.description = "Evaluate performance of pre-trained model in environment with increasing" \
                         "domain randomization"
    parser.add_argument('--fixed-difficulty', type=float, default=None,
                        help='Fix difficulty to specific value instead of '
                             'gradually increasing difficulty.')
    parser.add_argument('--num-augmentation-steps', type=int, default=100,
                        help='Number of augmentation steps to be performed. '
                             'The higher the number, the smaller the difference between two '
                             'experiments with regard to the grade of augmentation (degree of '
                             'domain randomization)')
    parser.add_argument('--num-runs-per-augmentation', type=int, default=100,
                        help='Number of runs per step on augmentation scale.'
                             'Results are averaged per step on scale.')
    parser.add_argument('--num-episodes-per-run', type=int, default=100,
                        help='Number of completed episodes each run should perform in the environment')
    parser.add_argument('--num-warmup-episodes', type=int, default=0,
                        help='Number of episodes which are performed before the actual evaluation in order '
                             'to let the environment warm-up, in order'
                             'to prevent wrong results')
    parser.add_argument('--use-randaugment', action="store_true",
                        help='If set, randaugment is used to augment images in evaluation')
    parser.add_argument('--randaugment-augmentations',
                        choices=RANDAUGMENT_MAP.keys(), nargs='+', type=str,
                        default=None,
                        help='Pass a list of augmentations to be used during evaluation')
    parser.add_argument('--randaugment-num-augmentations', type=int, default=3,
                        help='Number of augmentations used in Randaugment')
    parser.add_argument('--fixed-env-domain-randomization', type=float, default=None,
                        help='If set, the environment domain randomization is fixed to the given value (randaugment is not affected)')
    parser.add_argument('--table-surface', type=str,
                        help="Defines the table surface of the eval environment",
                        default='white')

    args = parser.parse_args(sys.argv[1:])
    use_tensorboard = args.use_tensorboard
    visualize = args.visualize
    log_base_dir = args.output_dir
    experiment_name = args.experiment_name if args.experiment_name else ""
    tag = "-" + args.tag if args.tag is not None else ""
    model_path = args.model_path
    seed = args.seed
    env_params_sampler_dict = import_env_params_from_file(
        args.env_params_file) if args.env_params_file else None
    num_warmup_episodes = args.num_warmup_episodes
    num_augmentation_steps = args.num_augmentation_steps
    num_runs_per_experiment = args.num_runs_per_augmentation
    num_episodes_per_run = args.num_episodes_per_run
    table_surface = args.table_surface

    fixed_env_domain_rand_amount = args.fixed_env_domain_randomization
    use_randaugment = args.use_randaugment
    randaugment_augs = args.randaugment_augmentations
    randaugment_num_augs = args.randaugment_num_augmentations

    fixed_difficulty = args.fixed_difficulty

    # Scale all experiments to a specific number of steps, so no matter how many steps we
    # evaluate, the results should be on the same scale.
    step_resolution = 1000

    # General setup
    date = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + "_seed-" + str(args.seed)
    run_name = date + "_exps-{}".format(num_augmentation_steps) + "_runs-{}".format(
        num_runs_per_experiment) + "_eps-{}".format(num_episodes_per_run) + tag
    log_dir = get_and_create_output_dir(log_base_dir, experiment_name, run_name)

    if use_tensorboard:
        tb_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))
        tb_writer.flush()

        with open(os.path.join(log_dir, "hyperparams.txt"), "w") as file:
            file.write("python " + " ".join(sys.argv) + "\n")
            for arg in vars(args):
                file.write(str(arg) + ' ' + str(getattr(args, arg)) + '\n')

    # Environment setup
    grasping_env = CurriculumEnvLog(task='stackVel', curr='no_dr', initial_pose='close',
                                    act_type='continuous', renderer='egl',
                                    obs_type='img_state_reduced', use_dr=False, max_steps=150,
                                    restitution=0.5, gripper_delay=12, img_type='rgb',
                                    adaptive_task_difficulty=True, table_surface=table_surface,
                                    position_error_obs=False, block_type='primitive', img_size="rl",
                                    movement_calib="old",
                                    env_params_sampler_dict=env_params_sampler_dict)
    experiment_env = build_env(grasping_env, normalize_obs=False)

    if use_randaugment:
        augmentation_list = setup_randaugment_augmentation_list(randaugment_augs)
        rand_aug = randaugment.RandAugment(num_augmentations=randaugment_num_augs, magnitude=0.0,
                                           augmentation_list=augmentation_list)
        transforms = augmenters.transforms_from_randaugment(rand_aug)
        experiment_env = AugmentationObservationWrapper(experiment_env, transforms=transforms)

    model_path = os.path.join(os.getcwd(), model_path)
    model = Model(experiment_env, model_path, deterministic=True)

    domain_randomization_rates = np.linspace(0, 1, num_augmentation_steps)
    random.seed(seed)
    seeds = [random.randint(0, sys.maxsize) for i in range(num_runs_per_experiment)]
    experiment_stat = {}
    obs_cache = {}

    # Perform some episodes in environment as warm-up to prevent wrong accuracy reports due
    # to bad performance after starting the environment
    for i in tqdm(range(num_warmup_episodes), desc="Warm Up"):
        # create variable / constant, this number is chosen empirically, as it took arround 5
        # 5 episodes to get stable results.
        grasping_env.seed(seed)

        warmup_obs, warmup_done = experiment_env.reset(), False
        warmup_env_step = 0
        while not warmup_done:
            warmup_action = model.step(warmup_obs, warmup_done)
            warmup_obs, warmup_rew, warmup_done, info = experiment_env.step(warmup_action)
            warmup_done = warmup_done.any() if isinstance(warmup_done,
                                                          np.ndarray) else warmup_done
            warmup_env_step += 1

    if fixed_difficulty is not None:
        num_augmentation_steps = 1

    for experiment_count in tqdm(range(num_augmentation_steps), desc="Experiment"):
        domain_rand_amount = domain_randomization_rates[experiment_count]
        experiment_step = int(domain_rand_amount * step_resolution)

        if fixed_difficulty is not None:
            domain_rand_amount = fixed_difficulty

        print("Running experiment {}/{} - Difficulty: {}".format(experiment_count,
                                                                 num_augmentation_steps - 1,
                                                                 domain_rand_amount))

        experiment_stat[experiment_count] = {
            "domain_randomization_amount": domain_rand_amount,
            "curriculum": None,
            "runs": [],
            "success_rate_per_run": [],
            "success_rate": None,
        }

        obs_cache[experiment_count] = []

        # Set domain_rand_amount
        env_domain_randomization = fixed_env_domain_rand_amount if fixed_env_domain_rand_amount else domain_rand_amount
        set_env_domain_randomization(grasping_env, env_domain_randomization)
        if use_randaugment:
            rand_aug.set_magnitude(domain_rand_amount)

        # Each experiment setup is repeated several times to achieve a better more accurate results
        for run_count in range(num_runs_per_experiment):
            run_episode_lengths = []
            run_successes = []
            run_rewards = []
            count_run_success_episodes = 0

            grasping_env.seed(seeds[run_count])

            for episode in range(num_episodes_per_run):
                experiment_env.eval = True

                obs = experiment_env.reset()
                env_step = 0
                done = False
                ep_rews = 0

                # Interact with the environment until episode is finished:
                #   either max number of steps reached or success
                while not done:
                    # Evaluate
                    action = model.step(obs, done)
                    obs, rew, done, info = experiment_env.step(action)
                    env_step += 1

                    # Unpack environment
                    done = done.any() if isinstance(done, np.ndarray) else done
                    is_episode_success = info[0]['task_success']
                    ep_rews += rew.cpu().flatten().numpy()[0]

                    if visualize:
                        render_obs(obs, sleep=1)

                    if env_step == 1 and run_count == 0:
                        # Store first obs of each episode  for visualization / logging it
                        obs_cache[experiment_count].append(obs.copy())

                    if done:
                        count_run_success_episodes += 1 if is_episode_success else 0
                        run_successes.append(is_episode_success)
                        run_episode_lengths.append(env_step)
                        run_rewards.append(ep_rews)
                        print("Episode {}/{} - Success: {}, Reward: {}, Steps: {}".format(
                            episode + 1, num_episodes_per_run, is_episode_success, ep_rews,
                            env_step))

            run_stats = {
                "run_success_rate": count_run_success_episodes / num_episodes_per_run,
                "run_episode_lengths": run_episode_lengths,
                "run_successes_per_episode": run_successes,
                "run_rewards_per_episode": run_rewards,
                "run_avg_reward": sum(run_rewards) / len(run_rewards),
                "run_avg_steps": sum(run_episode_lengths) / len(run_episode_lengths),
            }

            experiment_stat[experiment_count]["runs"].append(run_stats)

            print("Run {}/{} of experiment {} - Run Success Rate: {} - Avg Reward: {} - Seed: {}".format(
                run_count + 1, num_runs_per_experiment,
                experiment_count, run_stats["run_success_rate"], run_stats["run_avg_reward"],
                run_stats["run_avg_steps"]), seeds[run_count])

        # Calculate experiment statistics
        experiment_successes_per_run = [run["run_success_rate"] for run in
                                        experiment_stat[experiment_count]["runs"]]

        experiment_stat[experiment_count]["success_rate_per_run"] = experiment_successes_per_run
        experiment_stat[experiment_count]["success_rate"] = sum(
            experiment_successes_per_run) / len(
            experiment_successes_per_run)

        avg_reward_per_run = [run["run_avg_reward"] for run in
                              experiment_stat[experiment_count]["runs"]]
        experiment_stat[experiment_count]["avg_reward"] = sum(avg_reward_per_run) / len(
            avg_reward_per_run)

        avg_steps_per_run = [run["run_avg_steps"] for run in
                             experiment_stat[experiment_count]["runs"]]
        experiment_stat[experiment_count]["avg_steps"] = sum(avg_steps_per_run) / len(
            avg_steps_per_run)

        if use_tensorboard:
            tb_writer.add_scalar("success_rate_for_augmentations",
                                 experiment_stat[experiment_count]["success_rate"],
                                 experiment_step)
            tb_writer.add_scalar("avg_reward_for_augmentations",
                                 experiment_stat[experiment_count]["avg_reward"],
                                 experiment_step)
            tb_writer.add_scalar("avg_steps_for_augmentations",
                                 experiment_stat[experiment_count]["avg_steps"],
                                 experiment_step)
            # tb_writer.add_scalars()   # Use for plotting multiple scalars into one plot
            tb_writer.flush()

    if use_tensorboard:
        for step, obs in obs_cache.items():
            import torch

            obs_tensor = torch.cat([ob["img"] for ob in obs])
            tb_writer.add_images("Augmentations", obs_tensor / 255.0, step)

        with open(os.path.join(log_dir, "statistics_results.json"), "w") as file:
            json.dump(experiment_stat, file)

    print(json.dumps(experiment_stat))
    if use_tensorboard:
        tb_writer.close()
