import torch
from copy import deepcopy
import numpy as np

def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0]) 
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)

def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

class IMA():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def prepare_model_and_optimization(self, args):
        self.model.eval()
        self.scaler = torch.cuda.amp.GradScaler(init_scale=1000)

        trainable_param = [self.model.W]

        args.lr = 5e-3

        self.optimizer = torch.optim.AdamW(trainable_param, args.lr)
        self.optim_state = deepcopy(self.optimizer.state_dict())

    def pre_adaptation(self):
        with torch.no_grad():
            self.model.reset()
        
        self.optimizer.load_state_dict(self.optim_state)
    
    def adaptation_process(self, image, images, args):

        assert args.tta_steps > 0

        parts = args.algorithm.split("/")

        selected_idx = None
        coeff = None
        for j in range(args.tta_steps):
            with torch.cuda.amp.autocast():
                output = self.model(images)

                mid = parts[1]

                if (mid == 'all'):
                    entropys = softmax_entropy(output)

                    if coeff is None:
                        coeff = 1 / (torch.exp(((entropys.clone().detach()) - 0.4)))
                    
                    entropys = entropys.mul(coeff) 
                    loss = entropys.mean(0)
                
                if (mid == 'fil'):
                    if selected_idx is not None:
                        output = output[selected_idx]
                    else:
                        output, selected_idx = select_confident_samples(output, args.selection_p)

                    assert args.ent_loss or args.cont_loss

                    loss = avg_entropy(output)
                
                self.optimizer.zero_grad()

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()


        with torch.no_grad():
            with torch.cuda.amp.autocast():
                tta_output = self.model(image)
                tta_features, _, _ = self.model.forward_features(images)
                tta_outputs = self.model(images)

        return_dict = {
            'output':tta_output,
            'tta_features':tta_features,
            'tta_outputs':tta_outputs,
        }

        return return_dict