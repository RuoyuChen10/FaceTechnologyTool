import torch
import torch.nn.functional as F


class EvidenceLoss(torch.nn.Module):
    """Evidential Deep Learning"""

    def __init__(self, num_classes,
        total_epoch = 200,
        evidence = "relu",
        loss_type='log', 
        disentangle=False,
        with_avuloss = True,
        annealing_method='step', 
        annealing_start=0.01, 
        annealing_step=10
    ):
        super(EvidenceLoss, self).__init__()
        self.num_classes = num_classes
        self.evidence = evidence
        self.device = self.get_device()

        self.loss_type = loss_type
        self.disentangle = disentangle

        self.with_avuloss = with_avuloss

        self.annealing_method = annealing_method
        self.annealing_start = annealing_start
        self.annealing_step = annealing_step

        self.total_epoch = total_epoch

        self.eps = 1e-10

    def get_device(self):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda" if use_cuda else "cpu")
        return device

    def one_hot_embedding(self, labels):
        """Convert to One Hot Encoding
        """
        y = torch.eye(self.num_classes)
        return y[labels]

    def relu_evidence(self, y):
        """ReLU activation function
        """
        return F.relu(y)

    def exp_evidence(self, y):
        """EXP activation function
        """
        return torch.exp(torch.clamp(y, -10, 10))

    def softplus_evidence(self, y):
        """SoftPlus activation function
        """
        return F.softplus(y)

    def kl_divergence(self, alpha, num_classes):
        """KL loss, used for calibrate uncertainty
        """
        ones = torch.ones([1, num_classes], dtype=torch.float32, device=alpha.device)
        sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
        first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = (
            (alpha - ones)
            .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
            .sum(dim=1, keepdim=True)
        )
        kl = first_term + second_term
        return kl

    def loglikelihood_loss(self, y, alpha):
        y = y.to(self.device)
        alpha = alpha.to(self.device)
        S = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
        )
        loglikelihood = loglikelihood_err + loglikelihood_var
        return loglikelihood


    def mse_loss(self, y, alpha, epoch_num, num_classes, annealing_step, device=None):
        if not device:
            device = self.device
        y = y.to(device)
        alpha = alpha.to(device)
        loglikelihood = self.loglikelihood_loss(y, alpha, device=device)

        # annealing_coef = torch.min(
        #     torch.tensor(1.0, dtype=torch.float32),
        #     torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
        # )

        #kl_alpha = (alpha - 1) * (1 - y) + 1
        #kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
        return loglikelihood #+ kl_div

    def edl_mse_loss(self, output, target, epoch_num, num_classes, annealing_step, device=None):
        if not device:
            device = self.device
        evidence = self.relu_evidence(output)
        alpha = evidence + 1
        loss = torch.mean(
            self.mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
        )
        return loss

    def edl_log_loss(self, output, target, epoch_num, num_classes, annealing_step, device=None):
        if not device:
            device = self.device
        evidence = self.relu_evidence(output)
        alpha = evidence + 1
        loss = torch.mean(
            self.edl_loss(
                torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
            )
        )
        return loss

    def edl_digamma_loss(
        self, output, target, epoch_num, num_classes, annealing_step, device=None
    ):
        if not device:
            device = self.device
        evidence = self.relu_evidence(output)
        alpha = evidence + 1
        loss = torch.mean(
            self.edl_loss(
                torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
            )
        )
        return loss
    
    def compute_annealing_coef(self, **kwargs):
        if 'epoch' not in kwargs:
            epoch_num = self.total_epoch
        # assert 'epoch' in kwargs, "epoch number is missing!"
        # assert 'total_epoch' in kwargs, "total epoch number is missing!"
        # epoch_num, total_epoch = kwargs['epoch'], kwargs['total_epoch']
        else:
            epoch_num = kwargs['epoch']
        # annealing coefficient
        if self.annealing_method == 'step':
            annealing_coef = torch.min(torch.tensor(
                1.0, dtype=torch.float32), torch.tensor(epoch_num / self.annealing_step, dtype=torch.float32))
        elif self.annealing_method == 'exp':
            annealing_start = torch.tensor(self.annealing_start, dtype=torch.float32)
            annealing_coef = annealing_start * torch.exp(-torch.log(annealing_start) / self.total_epoch * epoch_num)
        else:
            raise NotImplementedError
        return annealing_coef

    def edl_loss(self, func, y, alpha, annealing_coef=0.5, target = None):
        y = y.to(self.device)
        alpha = alpha.to(self.device)
        S = torch.sum(alpha, dim=1, keepdim=True)

        A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

        if self.with_avuloss:
            pred_scores, pred_cls = torch.max(alpha / S, 1, keepdim=True)
            uncertainty = self.num_classes / S
            acc_match = torch.reshape(torch.eq(pred_cls, target.unsqueeze(1)).float(), (-1, 1))
            if self.disentangle:
                acc_uncertain = - torch.log(pred_scores * (1 - uncertainty) + self.eps)
                inacc_certain = - torch.log((1 - pred_scores) * uncertainty + self.eps)
            else:
                acc_uncertain = - pred_scores * torch.log(1 - uncertainty + self.eps)
                inacc_certain = - (1 - pred_scores) * torch.log(uncertainty + self.eps)
            avu_loss = annealing_coef * acc_match * acc_uncertain + (1 - annealing_coef) * (1 - acc_match) * inacc_certain
        # print("A: {}".format(A.shape))
        # print("avu_loss: {}".format(avu_loss.shape))
        # annealing_coef = torch.min(
        #     torch.tensor(1.0, dtype=torch.float32),
        #     torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
        # )
        # annealing_coef = 0.2

        # kl_alpha = (alpha - 1) * (1 - y) + 1
        # kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
        return A + avu_loss

    def forward(self, output, target, **kwargs):
        """Forward function.
        Args:
            output (torch.Tensor): The class score (before softmax).
            target (torch.Tensor): The ground truth label.
            epoch_num: The number of epochs during training.
        Returns:
            torch.Tensor: The returned EvidenceLoss loss.
        """
        results = {}

        # get evidence
        if self.evidence == 'relu':
            evidence = self.relu_evidence(output)
        elif self.evidence == 'exp':
            evidence = self.exp_evidence(output)
        elif self.evidence == 'softplus':
            evidence = self.softplus_evidence(output)
        else:
            raise NotImplementedError
        alpha = evidence + 1

        # one-hot embedding for the target
        y = self.one_hot_embedding(target)
        y = y.to(self.device)
        # compute annealing coefficient 
        annealing_coef = self.compute_annealing_coef(**kwargs)

        # compute the EDL loss
        if self.loss_type == 'mse':
            loss = self.mse_loss(y, alpha, annealing_coef)
        elif self.loss_type == 'log':
            loss = self.edl_loss(torch.log, y, alpha, annealing_coef, target)
        elif self.loss_type == 'digamma':
            loss = self.edl_loss(torch.digamma, y, alpha, annealing_coef, target)
        elif self.loss_type == 'cross_entropy':
            loss = self.ce_loss(target, y, alpha, annealing_coef)
        else:
            raise NotImplementedError

        # compute uncertainty and evidence
        _, preds = torch.max(output, 1)
        match = torch.reshape(torch.eq(preds, target).float(), (-1, 1))
        uncertainty = self.num_classes / torch.sum(alpha, dim=1, keepdim=True)
        total_evidence = torch.sum(evidence, 1, keepdim=True)
        evidence_succ = torch.sum(total_evidence * match) / torch.sum(match + 1e-20)
        evidence_fail = torch.sum(total_evidence * (1 - match)) / (torch.sum(torch.abs(1 - match)) + 1e-20)

        results["loss"] = loss
        results["uncertainty"] = uncertainty
        results["evidence_succ"] = evidence_succ
        results["evidence_fail"] = evidence_fail

        return loss.mean()
        

