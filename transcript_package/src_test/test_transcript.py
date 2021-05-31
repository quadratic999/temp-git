import pytest
import pandas as pd
from transcript import Transcript

class TestTranscript():
    
def test_prepare_calculate_rms(mocker):
    ss_wav_paths = ['/content/drive/MyDrive/workspace/transcript_package/src_test/wav_1.mp4', '/content/drive/MyDrive/workspace/transcript_package/src_test/wav_2.mp4']
    wav_path_origin = '/content/drive/MyDrive/workspace/transcript_package/src_test/wav_1.mp4'
    ctm_path = '/content/drive/MyDrive/workspace/transcript_package/src_test/ctm_example'

    # p = Mock_weather()
    transcript = Transcript(ss_wav_paths, wav_path_origin, ctm_path)
    moke_value = {'result': "雪", 'status': '下雪了！'}
    # 通过object的方式进行查找需要mock的对象
    p.weather = mocker.patch.object(Mock_weather, "weather", return_value=moke_value)
    result =p.weather_result()
    assert result=='下雪了！'


def test_get_transcript_rms():
    ss_wav_paths = ['/content/drive/MyDrive/workspace/transcript_package/src_test/wav_1.mp4', '/content/drive/MyDrive/workspace/transcript_package/src_test/wav_2.mp4']
    wav_path_origin = '/content/drive/MyDrive/workspace/transcript_package/src_test/wav_1.mp4'
    ctm_path = '/content/drive/MyDrive/workspace/transcript_package/src_test/ctm_example'
    transcript = Transcript(ss_wav_paths, wav_path_origin, ctm_path)
    result_df = transcript.get_transcript_rms()
    assert isinstance(result_df, pd.DataFrame)
