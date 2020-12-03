import yaml
import torch
import lib.models as models
import lib.datasets as datasets


class Config:
    def __init__(self, config_path):
        self.config = {}
        self.config_str = ""
        self.load(config_path)

    def load(self, path):
        with open(path, 'r') as file:
            self.config_str = file.read()
        self.config = yaml.load(self.config_str, Loader=yaml.FullLoader)

    def __repr__(self):
        return self.config_str

    def get_dataset(self, split):
        return getattr(datasets,
                       self.config['datasets'][split]['type'])(**self.config['datasets'][split]['parameters'])

    def get_model(self, **kwargs):
        name = self.config['model']['name']
        parameters = self.config['model']['parameters']
        return getattr(models, name)(**parameters, **kwargs)

    def get_optimizer(self, model_parameters):
        return getattr(torch.optim, self.config['optimizer']['name'])(model_parameters,
                                                                      **self.config['optimizer']['parameters'])

    def get_lr_scheduler(self, optimizer):
        return getattr(torch.optim.lr_scheduler,
                       self.config['lr_scheduler']['name'])(optimizer, **self.config['lr_scheduler']['parameters'])

    def get_loss_parameters(self):
        return self.config['loss_parameters']

    def get_train_parameters(self):
        return self.config['train_parameters']

    def get_test_parameters(self):
        return self.config['test_parameters']

    def __getitem__(self, item):
        return self.config[item]

    def __contains__(self, item):
        return item in self.config
