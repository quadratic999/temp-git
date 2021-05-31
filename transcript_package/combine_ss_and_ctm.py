import copy
from difflib import SequenceMatcher
from collections import defaultdict

import numpy as np
import pandas as pd
import librosa
from scipy.io import wavfile
from python_speech_features import mfcc


class Transcript():
    def __init__(self, ss_wav_paths, wav_path_origin, ctm_path):
        self.ss_wav_paths = ss_wav_paths
        self.wav_path_origin = wav_path_origin
        self.ctm_path = ctm_path

    def _ctm_to_groupdata(data_path, group_name):
        # load stt results
        ctm_df = pd.read_csv(data_path, sep=' ', header=None)
        ctm_df = ctm_df.iloc[:, :-1]
        ctm_df.columns = ['file_name', '1', 'start_time', 'stay_time', 'word']
        selectors = ctm_df.groupby('file_name')
        group_data = selectors.get_group(group_name)
        group_data = group_data.reset_index(drop=True)
        return group_data

    def get_times(group_data):
        times = []
        for i in group_data.index:
            start = group_data.loc[i, 'start_time']
            end = (start + group_data.loc[i, 'stay_time']).round(2)
            if times:
                if start == times[-1][1]:
                    last_start, last_end = times.pop()
                    start = last_start
            times.append((start, end))
        return times

    def _prepare_calculate_rms(wav_path, rate):
        y, _ = librosa.load(wav_path, sr=rate)
        rms_sample = librosa.feature.rms(y=y)
        _max, _min = np.amax(rms_sample), np.amin(rms_sample)
        return y, _max, _min 

    def get_sentences_rms(self, rate=8000):
        origin_ctm = _ctm_to_groupdata(self.ctm_path, self.wav_path_origin)

        template = {'sentences': list(), 'starts': list(), 'ends': list(), 
               'seg': list(), 'time': list()}
        
        transcript_para = defaultdict()
        for i, ss_wav_path in enumerate(self.ss_wav_paths):
            transcript_para[i] = copy.deepcopy(template)
            y, y_max, _min = _prepare_calculate_rms(ss_wav_path, rate)
            transcript_para[i]['y'] = y
            transcript_para[i]['max'] = _max
            transcript_para[i]['min'] = _min

        # 產出行員與顧客之個人逐字稿
        for i in origin_ctm.index:
            start = origin_ctm.loc[i, 'start_time']
            end = (start + origin_ctm.loc[i, 'stay_time']).round(2)
            text = origin_ctm.loc[i, 'word']
            if text == '<SIL>':
                continue

            rms_mean_lst = []
            for k, v in transcript_para.items():
                rms = librosa.feature.rms(y=v['y'][int(start*rate): int(end*rate)])
                rms = (rms - v['min']) / (v['max'] - v['min'])
                rms_lst.append(np.mean(rms))

            max_rms = max(rms_lst)
            rms_state_lst = [1 if rms == max_rms else 0 for rms in rms_lst]                      

            for j, rms_state in enumerate(rms_state_lst):
                if rms_state == 1:
                    transcript_para[j]['seg'] += [text]
                    transcript_para[j]['time'] += [start, end]                
                elif transcript_para[j]['seg']:
                    transcript_para[j]['sentences'].append(' '.join(a_seg))
                    transcript_para[j]['starts'].append(transcript_para[j]['time'][0])
                    transcript_para[j]['ends'].append(transcript_para[j]['time'][-1]) 
                    transcript_para[j]['seg'], transcript_para[j]['time'] = [], []

        # 依照發言時長，由長到短排定語者編號
        times = []
        for k, v in transcript_para.items():
            total_time = sum([end - start for start, end in zip(v['starts'], v['ends'])])
            times.append((k, total_time))
        times.sort(reverse = True, key = lambda s: s[1])
        
        df = pd.DataFrame(columns=['speaker', 'start', 'end', 'sentence'})
        for i, (index, _) in enumerate(times):
            df = df.append({'speaker': [i] * len(transcript_para[index]['starts']), 
                     'start': transcript_para[index]['starts'], 
                     'end': transcript_para[index]['ends'], 
                     'sentence': transcript_para[index]['sentences']})
        return df

    def get_transcript(self):
        


def get_report(customer_ctm, origin_ctm, questions, base_score=0.4):
    # find the question  time span (if score > base_score)
    q_result = [get_time_span(origin_ctm, q) for q in questions]
    q_result = sorted(q_result, key=lambda x: x['start_time'])

    # record  time info
    temp_record = []
    for i, qs in enumerate(q_result):
        # add question and time start & end
        temp_record.append([qs['start_time'], qs['end_time'], qs['max_score'], qs['ori_question'], ''.join(qs['tokens_text'])])

    for i in range(len(temp_record)):
        q_end = q_result[i]['end_time']
#         next_q_start = q_result[i]['end_time']
        sentences = group_2_asr.loc[customer_ctm['start'].apply(lambda x: q_end-2 < x < q_end+5), :]
        ans = sentences.values[0].tolist() if not sentences.empty else None
        if ans and (temp_record[i][2] != 0):
            ans.append(''.join(ans.pop().split()))
            temp_record[i] += ans
        else:
            temp_record[i] += [0, 0, None]
    report = pd.DataFrame(temp_record, columns=["q_start_time", "q_end_time", "score", "question", "recognize_result",
                                                "reply_start_time", "reply_end_time", "reply"])
    return report        

if __name__=='__main__':
    wav_name = '電訪-傳統型台幣-中壽'
    origin_name = f'/home/jovyan/wm-insur-call-qa/eric/speaker-separation/test_zone/test_result/{wav_name}.wav'
    origin_ctm = ctm_to_groupdata('/home/jovyan/exchanging-pool/to_owen/func_asr/stt_result/ctm/ctm', origin_wav_path)
    
    result_df = get_sentences_rms(origin_ctm, wav_path_a, wav_path_b)

    questions = ["""您好！這裡是玉山銀行總行個金處/OO分行/OO消金中心，
               敝姓O，員工編號OOOOO，請問是○○○先生/小姐本人嗎？""",
            '感謝您近期透過本行投保○○人壽○○○，繳費年期為O年，依照保險法令的要求，為保障您的權益，稍後電話訪問內容將會全程錄音，請問您同意嗎？'
            '為維護您的資料安全，這裡簡單跟您核對基本資料，您的身分證字號是，請問後三碼是？',
            '請問您的出生年月日是?',
            '請問您是否知道本次購買的是○○人壽的保險，不是存款，如果辦理解約將可能只領回部分已繳保費？',
            '請問您投保時，是否皆由○○消金中心的○○○，在旁邊協助，並由您本人○○○親自簽名，且被保險人之健康告知事項皆由您確認後親自填寫？',
            '請問○○消金中心的○○○是否有向您說明產品內容，並確認符合您的需求？',
            '請問招攬人員是否有提供您一次繳清與分期繳等不同繳費方式選擇？',
            '請問您本次投保繳交保費的資金來源是否為',
            """請問您是否已事先審慎評估自身財務狀況與風險承受能力，
               並願承擔因財務槓桿操作方式所面臨的風險及辦理保單解約轉投保之權益損失，
               除辦理貸款或保單借款需支付本金及利息外，
               還有該產品可能發生之相關風險及最大可能損失，
               且本行人員並未鼓勵或勸誘以辦理貸款、保單借款、保單解約/保單終止及定存解約之方式購買保險，
               請問您是否已瞭解？""",
            '與您確認，本保單之規劃您是否已確實瞭解投保目的、保險需求，並經綜合考量財務狀況以及付費能力，且不影響您的日常支出？',
            '與您再次確認上述投保內容和本次貸款並沒有搭售或不當行銷的情形發生，請問是否正確?',
            '請問您本次辦理貸款及保險，是否有新申請玉山網路銀行？']    
    
    get_report(customer_ctm, origin_ctm, questions)
