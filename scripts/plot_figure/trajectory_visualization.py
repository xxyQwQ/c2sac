import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp')
os.environ.setdefault('MESA_SHADER_CACHE_DIR', '/tmp/mesa_shader_cache')

import matplotlib.pyplot as plt
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.c2sac import C2SACAgent
from utils.env import make


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render image sequences for a trained C2SAC policy.'
    )
    parser.add_argument(
        '--checkpoint',
        type=Path,
        default=Path('checkpoints/c2sac/all-medium/seed-0/model.pth'),
        help='Path to the saved C2SAC model checkpoint.'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('figures'),
        help='Directory where rendered frames will be written.'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=300,
        help='Maximum environment steps to render per task.'
    )
    parser.add_argument(
        '--height',
        type=int,
        default=480,
        help='Rendered image height.'
    )
    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Rendered image width.'
    )
    parser.add_argument(
        '--camera-id',
        type=int,
        default=0,
        help='Camera id used for MuJoCo rendering.'
    )
    parser.add_argument(
        '--save-every',
        type=int,
        default=10,
        help='Save one rendered image every N environment steps.'
    )
    parser.add_argument(
        '--device',
        choices=('auto', 'cpu', 'cuda'),
        default='auto',
        help='Device used for policy inference.'
    )
    return parser.parse_args()


def resolve_device(device_name):
    if device_name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device_name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested, but no CUDA device is available.')
    return torch.device(device_name)


def load_config_from_checkpoint(model_path):
    config_path = model_path.parent / '.hydra' / 'config.yaml'
    if not config_path.is_file():
        raise FileNotFoundError(f'Could not find Hydra config next to checkpoint: {config_path}')
    return OmegaConf.load(config_path)


def build_agent(config, device):
    parameter = dict(config.agent.parameter)
    agent = C2SACAgent(
        num_tasks=len(config.setting.task_names),
        state_dims=config.setting.state_dims,
        action_dims=config.setting.action_dims,
        **parameter
    )
    agent.to_device(device)
    weights = torch.load(str(config.checkpoint_path), map_location=device)
    for name, module in agent.modules.items():
        module.load_state_dict(weights[name])
    return agent


def save_frame(image, frame_path):
    plt.imsave(frame_path, image)


def render_task_sequence(
    agent,
    device,
    task_name,
    task_index,
    seed,
    output_dir,
    max_steps,
    height,
    width,
    camera_id,
    save_every,
):
    env = make(f'walker_{task_name}', seed=seed)
    episode_dir = output_dir / task_name
    episode_dir.mkdir(parents=True, exist_ok=True)

    time_step = env.reset()
    frame = env.physics.render(height=height, width=width, camera_id=camera_id)
    save_frame(frame, episode_dir / 'frame_0000.png')

    total_reward = 0.0
    num_steps = 0
    task = torch.tensor([task_index], dtype=torch.long, device=device)

    while not time_step.last() and num_steps < max_steps:
        state = torch.from_numpy(time_step.observation).to(device)
        action = agent.take_action(task, state).cpu().numpy()
        time_step = env.step(action)

        num_steps += 1
        total_reward += float(time_step.reward)
        if num_steps % save_every == 0 or time_step.last() or num_steps == max_steps:
            frame = env.physics.render(height=height, width=width, camera_id=camera_id)
            save_frame(frame, episode_dir / f'frame_{num_steps:04d}.png')

    return {
        'task': task_name,
        'frames': num_steps + 1,
        'steps': num_steps,
        'return': total_reward,
        'output_dir': episode_dir,
    }


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    config = load_config_from_checkpoint(checkpoint_path)
    config.checkpoint_path = str(checkpoint_path)

    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    agent = build_agent(config, device)

    print(f'Loaded checkpoint: {checkpoint_path}')
    print(f'Using device: {device}')
    print(f'Saving frames under: {output_dir}')

    for task_index, task_name in enumerate(config.setting.task_names):
        summary = render_task_sequence(
            agent=agent,
            device=device,
            task_name=str(task_name),
            task_index=task_index,
            seed=int(config.seed),
            output_dir=output_dir,
            max_steps=args.max_steps,
            height=args.height,
            width=args.width,
            camera_id=args.camera_id,
            save_every=args.save_every,
        )
        print(
            f"Rendered task={summary['task']} "
            f"steps={summary['steps']} frames={summary['frames']} "
            f"return={summary['return']:.2f} dir={summary['output_dir']}"
        )


if __name__ == '__main__':
    main()
