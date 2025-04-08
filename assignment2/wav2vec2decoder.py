from typing import List, Tuple

import kenlm
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import Levenshtein
import heapq


class Wav2Vec2Decoder:
    def __init__(
        self,
        model_name="facebook/wav2vec2-base-960h",
        lm_model_path="lm/3-gram.pruned.1e-7.arpa",
        beam_width=3,
        alpha=1.0,
        beta=1.0,
    ):
        """
        Initialization of Wav2Vec2Decoder class

        Args:
            model_name (str): Pretrained Wav2Vec2 model from transformers
            lm_model_path (str): Path to the KenLM n-gram model (for LM rescoring)
            beam_width (int): Number of hypotheses to keep in beam search
            alpha (float): LM weight for shallow fusion and rescoring
            beta (float): Word bonus for shallow fusion
        """
        # once logits are available, no other interactions with the model are allowed
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)

        # you can interact with these parameters
        self.vocab = {i: c for c, i in self.processor.tokenizer.get_vocab().items()}
        self.blank_token_id = self.processor.tokenizer.pad_token_id
        self.word_delimiter = self.processor.tokenizer.word_delimiter_token
        self.beam_width = beam_width
        self.alpha = alpha
        self.beta = beta
        self.lm_model = kenlm.Model(lm_model_path) if lm_model_path else None

    def greedy_decode(self, logits: torch.Tensor) -> str:
        """
        Perform greedy decoding (find best CTC path)

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V)

        Returns:
            str: Decoded transcript
        """

        token_indices = torch.argmax(logits, dim=-1)

        # Уберем дубликаты, которые выдает CTC при предсказании
        collapsed_tokens = []
        prev_token = None
        for token in token_indices:
            if token != prev_token:
                collapsed_tokens.append(token.item())
                prev_token = token

        # Уберем blank токены
        filtered_tokens = [
            token for token in collapsed_tokens if token != self.blank_token_id
        ]

        # Переведем в строку
        transcript = "".join(
            [
                self.vocab[token].upper() if token in self.vocab else ""
                for token in filtered_tokens
            ]
        )

        # И наконец заменем разделители слов пробелами
        if self.word_delimiter in transcript:
            transcript = transcript.replace(self.word_delimiter, " ")

        return transcript

    def beam_search_decode(self, logits: torch.Tensor, return_beams: bool = False):
        """
        Perform beam search decoding (no LM)

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size
            return_beams (bool): Return all beam hypotheses for second pass LM rescoring

        Returns:
            Union[str, List[Tuple[float, List[int]]]]:
                (str) - If return_beams is False, returns the best decoded transcript as a string.
                (List[Tuple[List[int], float]]) - If return_beams is True, returns a list of tuples
                    containing hypotheses and log probabilities.
        """
        T, V = logits.shape
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        beam_width = self.beam_width

        # Инициируем список гипотез (log_prob, token_sequence)
        hypotheses = [([], 0.0)]

        for t in range(T):  # Для каждого временного шага
            new_hypotheses = []
            for seq, log_p in hypotheses:
                # Вычислим топ-N токенов на текущем шаге
                topk_log_probs, topk_tokens = torch.topk(log_probs[t], k=beam_width)

                for token, token_log_p in zip(topk_tokens, topk_log_probs):
                    new_seq = seq.copy()
                    # Фильтруем
                    if token != self.blank_token_id and (
                        not new_seq or token != new_seq[-1]
                    ):
                        new_seq.append(token.item())
                    # Обновим скор (сумма log-вероятностей)
                    new_log_p = log_p + token_log_p.item()
                    new_hypotheses.append((new_seq, new_log_p))

            # Отсортируем и выберем топ-beam_width гипотез
            hypotheses = heapq.nlargest(
                self.beam_width, new_hypotheses, key=lambda x: x[1]
            )

        # Возвращаем лучшую гипотезу или все лучи
        if return_beams:
            return hypotheses
        else:
            best_seq = hypotheses[0][0]
            collapsed = []
            prev = None
            for token in best_seq:
                if token != prev:
                    collapsed.append(token)
                    prev = token
            filtered = [t for t in collapsed if t != self.blank_token_id]
            transcript = "".join([self.vocab[token].upper() for token in filtered])
            if self.word_delimiter in transcript:
                transcript = transcript.replace(self.word_delimiter, " ")
            return transcript

    def beam_search_with_lm(self, logits: torch.Tensor) -> str:
        """
        Perform beam search decoding with shallow LM fusion

        Args:
            logits (torch.Tensor): Logits from Wav2Vec2 model (T, V), where
                T - number of time steps and
                V - vocabulary size

        Returns:
            str: Decoded transcript
        """
        T, V = logits.shape
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        beam_width = self.beam_width
        alpha = self.alpha
        beta = self.beta

        hypotheses = [([], 0.0, "")]

        for t in range(T):
            new_hypotheses = []
            for token_seq, log_p, text_seq in hypotheses:
                # Вычислим топ-N токенов для эффективности
                topk_log_probs, topk_tokens = torch.topk(log_probs[t], k=beam_width)

                for token, token_log_p in zip(topk_tokens, topk_log_probs):
                    new_token_seq = token_seq.copy()
                    new_text_seq = text_seq

                    # Отфильтруем
                    if token != self.blank_token_id:
                        if not new_token_seq or token != new_token_seq[-1]:
                            char = self.vocab[token.item()].upper()
                            new_token_seq.append(token.item())
                            new_text_seq += char

                    # Обновим скор (сумма log-вероятностей)
                    new_log_p = log_p + token_log_p.item()

                    # Получаем текст для LM (заменяем разделитель на пробел)
                    lm_text = new_text_seq.replace(self.word_delimiter, " ")
                    lm_score = self.lm_model.score(lm_text, bos=False, eos=False)
                    word_count = len(lm_text.split())

                    # Общий score с учетом LM
                    new_score = new_log_p + alpha * lm_score + beta * word_count
                    new_hypotheses.append((new_token_seq, new_score, new_text_seq))

            # Выбираем топ-beam_width гипотез
            hypotheses = heapq.nlargest(
                self.beam_width, new_hypotheses, key=lambda x: x[1]
            )

        # Возвращаем лучшую гипотезу
        best_text = hypotheses[0][2]
        if self.word_delimiter in best_text:
            best_text = best_text.replace(self.word_delimiter, " ")
        return best_text

    def lm_rescore(self, beams: List[Tuple[List[int], float]]) -> str:
        """
        Perform second-pass LM rescoring on beam search outputs

        Args:
            beams (list): List of tuples (hypothesis, log_prob)

        Returns:
            str: Best rescored transcript
        """
        rescored_beams = []

        for token_seq, log_prob in beams:
            # Преобразуем последовательность токенов в текст
            text = "".join([self.vocab[token].upper() for token in token_seq])

            # Заменяем разделитель слов на пробел (если нужно)
            if self.word_delimiter in text:
                text = text.replace(self.word_delimiter, " ")

            # Вычисляем LM скор
            lm_score = self.lm_model.score(text, bos=False, eos=False)

            word_count = len(text.split())
            # Комбинируем скор
            total_score = log_prob + self.alpha * lm_score + self.beta * word_count

            rescored_beams.append((text, total_score))

        # Сортируем по убыванию общего score
        rescored_beams.sort(key=lambda x: x[1], reverse=True)

        return rescored_beams[0][0]

    def decode(self, audio_input: torch.Tensor, method: str = "greedy") -> str:
        """
        Decode input audio file using the specified method

        Args:
            audio_input (torch.Tensor): Audio tensor
            method (str): Decoding method ("greedy", "beam", "beam_lm", "beam_lm_rescore"),
                where "greedy" is a greedy decoding,
                      "beam" is beam search without LM,
                      "beam_lm" is beam search with LM shallow fusion, and
                      "beam_lm_rescore" is a beam search with second pass LM rescoring

        Returns:
            str: Decoded transcription
        """
        inputs = self.processor(audio_input, return_tensors="pt", sampling_rate=16000)
        with torch.no_grad():
            logits = self.model(inputs.input_values.squeeze(0)).logits[0]

        if method == "greedy":
            return self.greedy_decode(logits)
        elif method == "beam":
            return self.beam_search_decode(logits)
        elif method == "beam_lm":
            return self.beam_search_with_lm(logits)
        elif method == "beam_lm_rescore":
            beams = self.beam_search_decode(logits, return_beams=True)
            return self.lm_rescore(beams)
        else:
            raise ValueError(
                "Invalid decoding method. Choose one of 'greedy', 'beam', 'beam_lm', 'beam_lm_rescore'."
            )


def test(decoder, audio_path, true_transcription):

    audio_input, sr = torchaudio.load(audio_path)
    assert sr == 16000, "Audio sample rate must be 16kHz"

    print("=" * 60)
    print("Target transcription")
    print(true_transcription)

    # Print all decoding methods results
    for d_strategy in ["greedy", "beam", "beam_lm", "beam_lm_rescore"]:
        print("-" * 60)
        print(f"{d_strategy} decoding")
        transcript = decoder.decode(audio_input, method=d_strategy)
        print(f"{transcript}")
        print(
            f"Character-level Levenshtein distance: {Levenshtein.distance(true_transcription, transcript.strip())}"
        )


if __name__ == "__main__":

    test_samples = [
        (
            "examples/sample1.wav",
            "IF YOU ARE GENEROUS HERE IS A FITTING OPPORTUNITY FOR THE EXERCISE OF YOUR MAGNANIMITY IF YOU ARE PROUD HERE AM I YOUR RIVAL READY TO ACKNOWLEDGE MYSELF YOUR DEBTOR FOR AN ACT OF THE MOST NOBLE FORBEARANCE",
        ),
        (
            "examples/sample2.wav",
            "AND IF ANY OF THE OTHER COPS HAD PRIVATE RACKETS OF THEIR OWN IZZY WAS UNDOUBTEDLY THE MAN TO FIND IT OUT AND USE THE INFORMATION WITH A BEAT SUCH AS THAT EVEN GOING HALVES AND WITH ALL THE GRAFT TO THE UPPER BRACKETS HE'D STILL BE ABLE TO MAKE HIS PILE IN A MATTER OF MONTHS",
        ),
        (
            "examples/sample3.wav",
            "GUESS A MAN GETS USED TO ANYTHING HELL MAYBE I CAN HIRE SOME BUMS TO SIT AROUND AND WHOOP IT UP WHEN THE SHIPS COME IN AND BILL THIS AS A REAL OLD MARTIAN DEN OF SIN",
        ),
        (
            "examples/sample4.wav",
            "IT WAS A TUNE THEY HAD ALL HEARD HUNDREDS OF TIMES SO THERE WAS NO DIFFICULTY IN TURNING OUT A PASSABLE IMITATION OF IT TO THE IMPROVISED STRAINS OF I DIDN'T WANT TO DO IT THE PRISONER STRODE FORTH TO FREEDOM",
        ),
        (
            "examples/sample5.wav",
            "MARGUERITE TIRED OUT WITH THIS LONG CONFESSION THREW HERSELF BACK ON THE SOFA AND TO STIFLE A SLIGHT COUGH PUT UP HER HANDKERCHIEF TO HER LIPS AND FROM THAT TO HER EYES",
        ),
        (
            "examples/sample6.wav",
            "AT THIS TIME ALL PARTICIPANTS ARE IN A LISTEN ONLY MODE",
        ),
        (
            "examples/sample7.wav",
            "THE INCREASE WAS MAINLY ATTRIBUTABLE TO THE NET INCREASE IN THE AVERAGE SIZE OF OUR FLEETS",
        ),
        (
            "examples/sample8.wav",
            "OPERATING SURPLUS IS A NON CAP FINANCIAL MEASURE WHICH IS DEFINED AS FULLY IN OUR PRESS RELEASE",
        ),
    ]

    decoder = Wav2Vec2Decoder()

    _ = [test(decoder, audio_path, target) for audio_path, target in test_samples]
