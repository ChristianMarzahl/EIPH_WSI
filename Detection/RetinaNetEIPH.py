from models.RetinaNet import *
from fastai.callbacks.hooks import num_features_model


class RetinaNetEIPH(RetinaNet):

    def __init__(self, encoder: nn.Module, n_classes, final_bias:float=0.,  n_conv:float=4,
                 chs=256, n_anchors=9, flatten=True, sizes=None):
        super().__init__(encoder, n_classes, final_bias, n_conv, chs, n_anchors, flatten, sizes)

        self.classifier_image = self._create_image_classifier(nf=num_features_model(self.encoder) * 2,
                                                              nc=1, y_range=[0-0.5, n_classes-0.5])

    def _create_image_classifier(self, nf:int, nc:int=1, y_range=[-0.5,4.5],
                           lin_ftrs:Optional[Collection[int]]=None, ps:Floats=0.5, bn_final:bool=False):
        "Model head that takes `nf` features, runs through `lin_ftrs`, and about `nc` classes."
        lin_ftrs = [nf, 512, nc] if lin_ftrs is None else [nf] + lin_ftrs + [nc]
        ps = listify(ps)
        if len(ps)==1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
        actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
        layers = [AdaptiveConcatPool2d(), Flatten()]
        for ni,no,p,actn in zip(lin_ftrs[:-1],lin_ftrs[1:],ps,actns):
            layers += bn_drop_lin(ni,no,True,p,actn)
        if bn_final: layers.append(nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
        if y_range is not None: layers.append(SigmoidRange(*y_range))
        return nn.Sequential(*layers)


    def forward(self, x):
        c5 = self.encoder(x)
        p_states = [self.c5top5(c5.clone()), self.c5top6(c5)]
        p_states.append(self.p6top7(p_states[-1]))
        for merge in self.merges:
            p_states = [merge(p_states[0])] + p_states
        for i, smooth in enumerate(self.smoothers[:3]):
            p_states[i] = smooth(p_states[i])
        if self.sizes is not None:
            p_states = [p_state for p_state in p_states if p_state.size()[-1] in self.sizes]
        return [self._apply_transpose(self.classifier, p_states, self.n_classes),
                self._apply_transpose(self.box_regressor, p_states, 4),
                [[p.size(2), p.size(3)] for p in p_states], self.classifier_image(c5)]