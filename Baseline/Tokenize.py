# ! apt-get install -y openjdk-8-jdk python3-dev
# ! pip install konlpy "tweepy<4.0.0"
# ! /bin/bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)

import konlpy
from typing import List, Tuple, Dict, Any
from Import_data import KoMRC

class TokenizedKoMRC(KoMRC):
    def __init__(self, data, indices: List[Tuple[int, int, int]]) -> None:
        super().__init__(data, indices)
        self._tagger = konlpy.tag.Mecab()
    
    def _tokenize_with_position(self, sentence: str) -> List[Tuple[str, Tuple[int, int]]]:
        position = 0
        tokens = []
        for morph in self._tagger.morphs(sentence):
            # 여기서의 position은 character 기반 position
            # 학습할 때는, token 기반의 position number가 필요하게 된다.
            position = sentence.find(morph, position)
            tokens.append((morph, (position, position + len(morph))))
            position += len(morph)
            # print(morph)
        
        return tokens
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = super().__getitem__(index)
        # sample의 keys -> guid, context, question, answers
        
        context, position = zip(*self._tokenize_with_position(sample['context']))
        context, position = list(context), list(position)
        question = self._tagger.morphs(sample['question'])
        
        # 캐릭터 단위로 저장되어 있는 position을 학습을 위해서 token 단위로 position 갱신
        # 그것을 위해 먼저 캐릭터 단위 position으로 확인했을 때, answer가 일치하는지 확인
        # 그 후에 position 새로 저장
        if sample['answers'] is not None:
            answers = []
            
            # 캐릭터별 index 상으로 일치하는 위치에 존재하는지 확인
            for answer in sample['answers']:
                for start, (position_start, position_end) in enumerate(position):
                    # print(position_start)
                    if position_start <= answer['answer_start'] < position_end:                        
                        break
                else:
                    # print(context, answer)
                    raise ValueError("No Matched start position!")
                
                # 같은 위치에 존재한다면, 정답이 똑같은 내용인지 확인
                target = "".join(answer['text'].split(" ")) # 기존 정답                
                source = "" # tokenized 한 후의 데이터에서 가져올 정답 저장할 변수
                for end, morph in enumerate(context[start :], start=start):
                    source += morph
                    if target in source:
                        break
                else:
                    # print(context, answer)
                    raise ValueError("No matched end position!")
                
                answers.append({
                    "start": start,
                    "end": end
                })
        else:
            answers = None
        
        return {
            "guid": sample['guid'],
            "original_context": sample["context"],
            "original_question": sample["question"],
            "tokenized_context": context,
            "tokenized_question": question,
            "context_position": position,
            "answers": answers
        }
        
# data = TokenizedKoMRC.load("/Users/chanwoo/Downloads/trainForMrc.json")
# train, val = TokenizedKoMRC.train_val_split(data)

# sample = train[4]
# print(sample['answers'])
# print(sample["tokenized_context"][sample["answers"][0]["start"] : sample["answers"][0]["end"] + 1])