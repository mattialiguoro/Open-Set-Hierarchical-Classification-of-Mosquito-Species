import gc

# !pip install timm
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.optim
import torchvision.models as models
from torch.nn import functional as F
from tqdm import tqdm

from utils.losses_metrics import History, MetricMonitor, calculate_accuracy


class Mosaico_net(nn.Module):
    def __init__(self, out_classes, dropout = 0, FE_type = "resnext50_32x4d"):
        super().__init__()

        #self.FE_type = FE_type
        if FE_type == "tf_efficientnetv2_s.in21k":
            self.feature_extractor = timm.create_model('tf_efficientnetv2_s.in21k', pretrained=True, num_classes=0)
            in_features = self.feature_extractor.num_features  # feature dim
        elif FE_type == "tf_efficientnetv2_m.in21k":
            self.feature_extractor = timm.create_model('tf_efficientnetv2_m.in21k', pretrained=True, num_classes=0)
            in_features = self.feature_extractor.num_features  # feature dim
        elif FE_type == "resnext50_32x4d":
            self.feature_extractor = models.resnext50_32x4d(weights='IMAGENET1K_V2')
            in_features = 1000

        self.classifier = nn.Sequential(
                                          nn.Dropout(p=dropout),
                                          nn.Linear(in_features,256),
                                          nn.ReLU(),
                                          nn.Dropout(p=dropout),
                                          nn.Linear(256,out_classes),
                                        )

    def forward(self,x):

        x_fe = self.feature_extractor(x)
        x_class = self.classifier(x_fe)
        x_f = F.softplus(x_class)

        return x_f


    def freeze_FE(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    def unfreeze_FE(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = True


# -
class MOSAICO_training:
    def __init__(self, params):

        self.params = params
        self.device = params["device"]
        self.training_method = self.params["training_method"]
        self.model = Mosaico_net(params["n_classes"], dropout = params["dropout"], FE_type = params["model_type"])
        self.model.freeze_FE()
        #self.criterion = nn.CrossEntropyLoss(reduction = "mean").to(self.device)#no focal loss GOODwin uses focal loss
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4 ,weight_decay=params["lambdaL2"])
        self.history = History()
        self.check_point_path = self.params["checkpoint"]

    def evidence_to_prob(self, evidence):
        alpha = evidence + 1
        if self.training_method == 'm-EDL':
            e_u = torch.full((evidence.shape[0], 1), evidence.shape[1], device=self.device)
            evidence_tot = torch.cat((evidence, e_u), dim=1)
            alpha = evidence_tot + 1
        strength = alpha.sum(dim=-1)
        p = alpha / strength[:, None]
        return p, alpha, strength

    def EDL_loss(self, evidence, target, epoch):
        p, alpha, strength = self.evidence_to_prob(evidence)
        one_hot_target = F.one_hot(target, num_classes=p.shape[1]).float()
        # calcolo i termini per MaximumLikelihood Loss
        err = ((one_hot_target - p) ** 2).sum(dim=-1)
        var = (p * (1 - p) / (strength[:, None] + 1)).sum(dim=-1)
        ML_loss = err + var
        # calcolo i termini per KL Loss
        alpha_tilde = one_hot_target + (1 - one_hot_target) * alpha
        first = (torch.lgamma(alpha_tilde.sum(dim=-1)) - torch.lgamma(alpha_tilde.new_tensor(float(evidence.shape[-1]))) \
                 - (torch.lgamma(alpha_tilde)).sum(dim=-1))
        second = ((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(alpha_tilde.sum(dim=-1))[:, None])).sum(dim=-1)
        KL_loss = first + second
        # Combino le due Loss
        lambda_t = min(1.0, epoch/10)
        final_loss = ML_loss + lambda_t*KL_loss
        return final_loss.mean()

    def validate(self, val_loader,epoch):
        metric_monitor = MetricMonitor()
        self.model.eval()
        stream = tqdm(val_loader)
        with torch.no_grad():
            for (images, target) in stream:
                images = images.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True).long()
                evidence = self.model(images)
                loss = self.EDL_loss(evidence, target, epoch)
                prob, _, _ = self.evidence_to_prob(evidence)
                accuracy = calculate_accuracy(prob, target)

                metric_monitor.update("Loss", loss.item())
                metric_monitor.update("Accuracy", accuracy)
                #metric_monitor.update("Reg", reg.item())
                stream.set_description(
                    "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
                )

        val_loss = metric_monitor.return_averages("Loss")
        val_acc = metric_monitor.return_averages("Accuracy")
        #val_reg = metric_monitor.return_averages("Reg")
        self.history.update("validation",val_loss,val_acc)
        if ((val_loss-self.history.best_val) < 0.001) or epoch==15:# smalles since loss#forse inutile# prova a vedere se tracciare la loss è meglio
            self.history.best_val = val_loss
            self.history.best_epoch = self.history.validation_loss["epoch"]
            print("Best epoch with LOSS: {:0.2f} - CKP saved".format(val_loss))
            torch.save(self.model.state_dict(), self.check_point_path)



    def train(self,train_loader, epoch):
        metric_monitor = MetricMonitor()
        self.model.train()
        stream = tqdm(train_loader)
        for (images, target) in stream:
            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True).long()
            evidence = self.model(images)
            loss = self.EDL_loss(evidence, target, epoch) 
            prob, _, _ = self.evidence_to_prob(evidence)
            accuracy = calculate_accuracy(prob, target)

            metric_monitor.update("Loss", loss.item())
            metric_monitor.update("Accuracy", accuracy)
            #metric_monitor.update("Reg", reg.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            stream.set_description(
                "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )

        train_loss = metric_monitor.return_averages("Loss")
        train_acc = metric_monitor.return_averages("Accuracy")
        #train_reg = metric_monitor.return_averages("Reg")
        self.history.update("train",train_loss,train_acc)
        self.history.update_lr(self.optimizer.param_groups[0]['lr'])
        #self.history.update_reg(reg.item())

    def do_train(self,train_loader,val_loader,epochs,lr,count,FREEZE=True):
        self.model = self.model.to(self.device)
        if FREEZE:
            self.model.freeze_FE()
        else:
            self.model.unfreeze_FE()

        self.optimizer.param_groups[0]['lr'] = lr
        for epoch in range(count+1, count+epochs + 1):
            count += 1
            self.train(train_loader, epoch)
            self.validate(val_loader, epoch)
        return count

    def test(self,val_loader,device="cpu"):

        model = Mosaico_net(self.params["n_classes"],FE_type = self.params["model_type"])
        model.load_state_dict(torch.load(self.check_point_path))

        model.eval()
        model.to(device)

        y_pred = []
        y_true = []
        incertezza = []
        with torch.no_grad():
            for (images, target) in val_loader:

                images = images.to(device)
                target = target.to(device).long()
                evidence = model(images)
                prob, _, strength = self.evidence_to_prob(evidence)
                if self.training_method == 'EDL':
                    u = prob.shape[1] / strength
                    incertezza.extend(u.cpu().numpy())

                y_pred.extend(prob.cpu().numpy())
                y_true.extend(target.cpu().numpy())

        if self.training_method == 'EDL':
            return np.array(y_pred), np.array(y_true),np.array(incertezza)
        elif self.training_method == 'm-EDL':
            return np.array(y_pred), np.array(y_true)

    def unload_model(self):
        self.model = self.model.to("cpu")
        del self.model
        torch.cuda.empty_cache()
        gc.collect()

    def load_model(self, path = ""):
        if path == "":
            path = self.check_point_path

        self.model = Mosaico_net(self.params["n_classes"],FE_type = self.params["model_type"])
        self.model.load_state_dict(torch.load(path))
