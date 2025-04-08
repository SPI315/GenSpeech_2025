# Assignment 2. ASR decoding - [20 pts]

In this exercise you are required to implement **4 ASR decoding methods** for a pre-trained CTC acoustic model [wav2vec2](https://huggingface.co/docs/transformers/en/model_doc/wav2vec2) for Speech Recognition in English:

1. Greedy decoding: [[line 41]](wav2vec2decoder.py#41)
2. Beam search decoding: [[line 54]](wav2vec2decoder.py#54)
3. Beam search with LM scores fusion: [[line 76]](wav2vec2decoder.py#76)
4. Beam search with a second pass LM rescoring: [[line 95]](wav2vec2decoder.py#95)


## Description

You are provided with:
- [`Wav2Vec2Decoder`](wav2vec2decoder.py) class where you have to implement the CTC decoding logic without importing additional libraries or other resources. Calling any method from `self.processor` and `self.model` is prohibited apart from the ones already implemented for receiving the **logits** matrix
    - Acoustic model [`facebook/wav2vec2-base-960h`](https://huggingface.co/facebook/wav2vec2-base-960h) was trained using 960 hours of training data from [LibriSpeech dataset](https://www.openslr.org/12)
- Pre-trained on LibriSpeech [3-gram KenLM language model](http://www.openslr.org/11/)
- Arbitrary [16 kHz audio files](examples/) in English language with [corresponding transcripts](wav2vec2decoder.py#165) for debugging

- Model vocabulary consists from the following characters
```python
{0: '<pad>', 1: '<s>', 2: '</s>', 3: '<unk>', 4: '|', 5: 'E', 6: 'T', 7: 'A', 8: 'O', 9: 'N', 10: 'I', 11: 'H', 12: 'S', 13: 'R', 14: 'D', 15: 'L', 16: 'U', 17: 'M', 18: 'W', 19: 'C', 20: 'F', 21: 'G', 22: 'Y', 23: 'P', 24: 'B', 25: 'V', 26: 'K', 27: "'", 28: 'X', 29: 'J', 30: 'Q', 31: 'Z'}
```
where

    - '<s>' - beginning of sentence token    <-- NOT USED IN THIS MODEL
    - '</s>' - end of sentence token         <-- NOT USED IN THIS MODEL
    - '<unk>' - unknown token                <-- NOT USED IN THIS MODEL
    - '<pad>' - blank symbol in CTC decoding
    - '|' - word separator token, interchangeable with ' ' (space) symbol in this exercise


In this assignment we don't focus on punctuation and text normalization - all ground truth transcripts are provided in **unnormalized uppercase characters**. The same is expected from the decoded transcription


## Installation

```bash
sudo sh -c 'apt-get update && apt-get upgrade && apt-get install cmake'
python3 -m pip install https://github.com/kpu/kenlm/archive/master.zip
python3 -m pip install levenshtein
```


## Tasks

1. Implement all required decoding methods and add to the report comparison of their quality (either in normalized Levenshtein distance or [WER](https://en.wikipedia.org/wiki/Word_error_rate))

2. To see the effect of LM model, try loading larger N-gram LM model from [link](http://www.openslr.org/11/) and report how results are changed for the test audios

3. Vary values of `beam_width`, `alpha` and `beta` parameters in all versions of beam search decoding and add your observations to the report


## Extra

Any other LM can be used for hypotheses rescoring (even BERT-based). In case you want to train your own N-gram [KenLM](https://github.com/kpu/kenlm) model, please refer to the [tutorials](#resources). Include your observations in the report if you've tried any of these approaches


## Evaluation

Your work will be tested against a small hold-out subset of audio files and CER & WER metrics will be reported


## Notes

- Don't forget to apply `torch.log_softmax()` to logits to get log probabilities
- Use python `heapq` module for storing most likely hypotheses during beam search


## Resources
- [DLA course CTC decoding lecture slides](https://docs.google.com/presentation/d/1cBXdNIbowwYNp42WhJmd1Pp85oeslOrKNmGyZa5HKBQ/edit?usp=sharing)
- [HuggingFace wav2vec2 tutorial with n-gram LMs](https://huggingface.co/blog/wav2vec2-with-ngram)
