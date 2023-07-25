# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MarsMapsDataset(BaseSegDataset):
    """Mars Daily Global Map dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for MDGM dataset.
    """

    METAINFO = dict(
        classes=("background", "dust"), palette=[[128, 128, 128], [98, 222, 98]]
    )

    def __init__(
        self,
        img_suffix=".png",
        seg_map_suffix=".png",
        reduce_zero_label=False,
        **kwargs
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs
        )
