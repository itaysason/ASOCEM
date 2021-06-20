import click
import warnings

warnings.filterwarnings('ignore')
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(chain=False, context_settings=CONTEXT_SETTINGS)
@click.option('--debug/--no-debug', default=False, help="Default is --no-debug.")
@click.option('-v', '--verbosity', default=0, help='Verbosity level (0-3).')
def simple_cli(debug, verbosity):
    return


@simple_cli.command('ASOCEM')
@click.option('--in_dir', type=str)
@click.option('--out_dir', type=str)
@click.option('--particle_size', type=int, default=300)
@click.option('--downsample_size', type=int, default=200)
@click.option('--window_size', type=int, default=3)
@click.option('--contamination_criterion', type=str, default='size')
@click.option('--algorithm', type=str, default='slow2')
@click.option('--n_cores', type=int, default=5)
def ASOCEM(in_dir, out_dir, particle_size, downsample_size, window_size, contamination_criterion, algorithm, n_cores):
    if 'regular' == algorithm:
        from src.ASOCEM_regular import ASOCEM_ver1
    elif 'fast' == algorithm:
        from src.ASOCEM_fast import ASOCEM_ver1
    else:
        raise ValueError('algorithm can only be regular or fast')
    ASOCEM_ver1(in_dir, out_dir, particle_size, downsample_size, window_size, contamination_criterion, n_mgraphs_sim=n_cores)


if __name__ == "__main__":
    simple_cli()
