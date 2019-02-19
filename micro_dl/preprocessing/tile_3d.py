import numpy as np
import os
import pandas as pd

from micro_dl.preprocessing import ImageTilerUniform
import micro_dl.utils.tile_utils as tile_utils
from micro_dl.utils import aux_utils as aux_utils


class ImageTilerUniform3D(ImageTilerUniform):
    """Tiles all volumes in a dataset"""

    def __init__(self,
                 input_dir,
                 output_dir,
                 tile_dict,
                 tile_size=[64, 64, 64],
                 step_size=[32, 32, 32],
                 depths=1,
                 time_ids=-1,
                 channel_ids=-1,
                 slice_ids=-1,
                 pos_ids=-1,
                 hist_clip_limits=None,
                 flat_field_dir=None,
                 isotropic=False,
                 image_format='zyx',
                 num_workers=4,
                 int2str_len=3):
        """Init

        Please ref to init of ImageTilerUniform.
        depth is not used here.
        """

        super().__init__(input_dir,
                         output_dir,
                         tile_dict,
                         tile_size,
                         step_size,
                         depths,
                         time_ids,
                         channel_ids,
                         slice_ids,
                         pos_ids,
                         hist_clip_limits,
                         flat_field_dir,
                         isotropic,
                         image_format,
                         num_workers,
                         int2str_len)

        assert len(self.tile_size) == 3, ''



