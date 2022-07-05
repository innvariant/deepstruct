import torch.nn as nn

from deepstruct.sparse import MaskableModule
from deepstruct.sparse import MaskedLinearLayer


def test_constructor_success():
    MaskableModule()


def test_inheritance__success():
    expected_child_unmaskable = nn.Linear(10, 5)
    expected_child_maskable = MaskedLinearLayer(10, 5)

    class InheritedMaskableModule(MaskableModule):
        def __init__(self):
            super().__init__()
            self._linear1 = expected_child_unmaskable
            self._linear2 = expected_child_maskable

    model = InheritedMaskableModule()

    assert expected_child_maskable in model.maskable_children
    assert expected_child_unmaskable not in model.maskable_children
