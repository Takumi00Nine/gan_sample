import click
from net import NetManager

@click.command()
@click.argument('model_name', type=str)
@click.option('--epoch', '-e', type=int, default=10)
@click.option('--input_dir', '-i', type=str, default='image')
@click.option('--output_dir', '-o', type=str, default='result')
def main(model_name, epoch, input_dir, output_dir):
    net_manager = NetManager(model_name, input_dir, output_dir)
    net_manager.init_model()
    for _ in range(epoch):
        net_manager.train()
        net_manager.test()
    net_manager.fina_model()

if __name__ == "__main__":
    main()
