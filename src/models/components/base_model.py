import torch
from omegaconf.dictconfig import DictConfig


class BaseModel(torch.nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.visuals = {}
        self.output = {}

    def forward(self, input):
        self.forward_start()
        self.forward_func(input)
        self.forward_end()
        return self.output, self.visuals

    def forward_start(self):
        self.visuals = {}
        self.output = {}

    def forward_func(self):
        raise NotImplementedError("forward function not implemented")

    def forward_end(self):
        self.update_visuals()
        assert self.output

    def grouped_parameters_with_lr(self):
        if "name" in self.hparams:
            network_name = self.__class__.__name__ + self.hparams.name
        else:
            network_name = self.__class__.__name__

        if "nonbase_layers" in self.hparams and len(self.hparams.nonbase_layers) > 0:
            assert (
                isinstance(self.hparams.nonbase_layers, DictConfig)
                and len(self.hparams.nonbase_layers) > 0
            )
            assert len(self.hparams.nonbase_layers) == 1
            counts = {}
            nonbase_params_dictlist = []
            for name, _ in self.named_parameters():
                counts[name] = 0
            for layer, dict in self.hparams.nonbase_layers.items():
                nonbase_params = []
                for name, params in self.named_parameters():
                    if name.startswith(layer):
                        nonbase_params.append(params)
                        counts[name] = counts[name] + 1

                if "grad_clip_val" in dict:
                    assert len(dict) == 2
                    nonbase_params_dictlist.append(
                        {
                            "params": nonbase_params,
                            "lr": dict["lr"],
                            "name": ".".join([network_name, layer]),
                            "grad_clip_val": dict["grad_clip_val"],
                        }
                    )
                else:
                    assert len(dict) == 1
                    nonbase_params_dictlist.append(
                        {
                            "params": nonbase_params,
                            "lr": dict["lr"],
                            "name": ".".join([network_name, layer]),
                        }
                    )

            base_params = []

            has_nonbase = False
            for name, params in self.named_parameters():
                assert name in counts.keys()
                if counts[name] == 0:
                    base_params.append(params)
                elif counts[name] == 1:
                    has_nonbase = True
                else:
                    assert False
            assert has_nonbase

            if "module_lr" in self.hparams and self.hparams.module_lr != None:
                base_params_dict_list = [
                    {
                        "params": base_params,
                        "name": ".".join([network_name, "base"]),
                        "lr": self.hparams["module_lr"],
                    }
                ]
            else:
                base_params_dict_list = [
                    {"params": base_params, "name": ".".join([network_name, "base"])}
                ]

            return base_params_dict_list + nonbase_params_dictlist

        else:
            if "module_lr" in self.hparams and self.hparams.module_lr != None:
                return [
                    {
                        "params": list(self.parameters()),
                        "name": network_name,
                        "lr": self.hparams["module_lr"],
                    }
                ]
            else:
                return [{"params": list(self.parameters()), "name": network_name}]

    def update_visuals(self):
        pass
