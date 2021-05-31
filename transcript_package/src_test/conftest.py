#!/usr/bin/env python
#coding=utf-8

import sys
sys.path.extend(['src', '/usr/local/lib/python3.7/dist-packages'])

import pytest
from transcript import Transcript

@pytest.fixture(scope = 'module')
def transcript():
    ss_wav_paths = ['/content/drive/MyDrive/workspace/transcript_package/src_test/wav_1.mp4', '/content/drive/MyDrive/workspace/transcript_package/src_test/wav_1.mp4']
    wav_path_origin = '/content/drive/MyDrive/workspace/transcript_package/src_test/wav_1.mp4'
    ctm_path = '/content/drive/MyDrive/workspace/transcript_package/src_test/ctm_example.csv'
    yield Transcript(ss_wav_paths, wav_path_origin, ctm_path)
