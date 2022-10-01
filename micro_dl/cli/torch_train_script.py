import yaml
import argparse

import micro_dl.torch_unet.utils.training as train


def read_config(config_path):
    '''
    One-line to safely open config files for argument reading
    
    :param str config_path: abs or relative path to configuration file
    
    :return dict config: a dictionary of config information read from input file
    '''
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args():
    """
    Parse command line arguments
    In python namespaces are implemented as dictionaries
    
    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        help='path to yaml configuration file',
    )
    parser.add_argument(
        '--gpu',
        type=str,
        help='intended gpu device number',
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    torch_config = read_config(args.config)
    network_config = torch_config['model']
    training_config = torch_config['training']
    
    #Instantiate training object
    trainer = train.TorchTrainer(torch_config)
    
    #generate dataloaders and init model
    trainer.generate_dataloaders()
    trainer.load_model()
    
    #train
    trainer.train()
