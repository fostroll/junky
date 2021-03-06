# -*- coding: utf-8 -*-
# junky lib: dataset
#
# Copyright (C) 2020-present by Sergei Ternovykh, Anastasiya Nikiforova
# License: BSD, see LICENSE for details
"""
Provides implementagions of torch.utils.data.Dataset for different purposes.
"""
from junky.dataset.dummy_dataset import *
from junky.dataset.len_dataset import *
from junky.dataset.label_dataset import *

from junky.dataset.word_dataset import *
from junky.dataset.bert_dataset import *
from junky.dataset.bert_tokenized_dataset import *

from junky.dataset.char_dataset import *
from junky.dataset.token_dataset import *

from junky.dataset.frame_dataset import *
from junky.dataset.word_cat_dataset import *
