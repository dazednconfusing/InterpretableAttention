from typing import Dict

import torch
import torch.nn as nn
from allennlp.common import Params
from allennlp.common.from_params import FromParams
from Transparency.model.modelUtils import BatchHolder, BatchMultiHolder, isTrue
from Transparency.model.modules.Attention import Attention, masked_softmax

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AttnDecoder(nn.Module, FromParams):
    def __init__(
        self,
        hidden_size: int,
        attention: Dict,
        output_size: int = 1,
        use_attention: bool = True,
        regularizer_attention: Dict = None,
    ):
        super().__init__()
        attention["hidden_size"] = hidden_size
        self.attention = Attention.from_params(Params(attention)).to(device)

        self.hidden_size = self.attention.hidden_size
        self.output_size = output_size
        self.linear_1 = nn.Linear(self.hidden_size, output_size)

        self.use_regulariser_attention = False
        if regularizer_attention is not None:
            regularizer_attention["hidden_size"] = self.hidden_size
            self.regularizer_attention = Attention.from_params(
                Params(regularizer_attention)
            )
            self.use_regulariser_attention = True

        self.use_attention = use_attention

    def randomize_attn(self):
        """All attention weights are redrawn from a
        gaussian with the same mean and std dev as the non-randomized distribution.

        Author: Sebastian Peralta
        """
        self.attention.randomize_weights()

    def decode(self, predict):
        predict = self.linear_1(predict)
        return predict

    def forward(self, data: BatchHolder):

        if self.use_attention:
            output = data.hidden
            mask = data.masks
            attn = self.attention(data)

            if self.use_regulariser_attention:
                data.reg_loss = 5 * self.regularizer_attention.regularise(
                    data.seq, output, mask, attn
                )

            if isTrue(data, "detach"):
                attn = attn.detach()

            # Author: Sebastian Peralta
            if isTrue(data, "permute"):
                permutation = data.generate_permutation()
                attn = torch.gather(attn, -1, torch.LongTensor(permutation).to(device))

            context = (attn.unsqueeze(-1) * output).sum(1)
            data.attn = attn
        else:
            context = data.last_hidden

        predict = self.decode(context)
        data.predict = predict

    def get_attention(self, data: BatchHolder):
        output = data.hidden_volatile
        mask = data.masks
        attn = self.attention(data)
        data.attn_volatile = attn

    def get_output(self, data: BatchHolder):
        output = data.hidden_volatile  # (B, L, H)
        attn = data.attn_volatile  # (B, *, L)

        if len(attn.shape) == 3:
            context = (attn.unsqueeze(-1) * output.unsqueeze(1)).sum(2)  # (B, *, H)
            predict = self.decode(context)
        else:
            context = (attn.unsqueeze(-1) * output).sum(1)
            predict = self.decode(context)

        data.predict_volatile = predict

    def get_output_from_logodds(self, data: BatchHolder):
        attn_logodds = data.attn_logodds  # (B, L)
        attn = masked_softmax(attn_logodds, data.masks)

        data.attn_volatile = attn
        data.hidden_volatile = data.hidden

        self.get_output(data)
