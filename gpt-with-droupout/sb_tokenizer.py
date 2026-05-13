# Copy of production_bpe_tokenizer.py
# Removed the comments
from collections import Counter
from typing import Dict, List
import regex as re


class SBTokenizer:
    def __init__(self):

            self.pattern = re.compile(
            r"""'(?:s|t|re|ve|m|ll|d)|"""  # Contractions: 's, 't, 're, 've, 'm, 'll, 'd
            r""" ?\p{L}+|"""                # Optional space + letters (words)
            r""" ?\p{N}+|"""                # Optional space + numbers
            r""" ?[^\s\p{L}\p{N}]+|"""     # Optional space + punctuation/symbols
            r"""\s+(?!\S)|"""               # Whitespace (not followed by non-whitespace)
            r"""\s+"""                      # Remaining whitespace
        )

            self.byte_encoder = self._bytes_to_unicode()
            self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
       
            self.vocab = {char: idx for idx, char in enumerate(self.byte_encoder.values())}
            self.reverse_vocab = {}  # token_id → token mapping (for decode)
            self.merges = {}  # (pair_a, pair_b) → merged_token mapping
            self.merge_ranks = {}  # merged_token → rank (order of merge)

    
            self.bpe_cache = {}  # token → List[tokens] mapping

            self.special_tokens = {}
            self.special_tokens_pattern = None
    

    def _bytes_to_unicode(self):
        # Start with printable ASCII range (excluding control characters and space)
        # 33-126 are printable: !"#$%&'()*+,-./0-9:;<=>?@A-Z[\]^_`a-z{|}~
        bs = list(range(ord("!"), ord("~") + 1))

        # Add Latin-1 supplement printable characters
        # 161-172: ¡¢£¤¥¦§¨©ª«¬
        bs.extend(list(range(ord("¡"), ord("¬") + 1)))

        # 174-255: ®¯°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ
        bs.extend(list(range(ord("®"), ord("ÿ") + 1)))

        # cs will be the Unicode characters we map to
        cs = bs.copy()

        # Now add mappings for the "problematic" bytes (control chars, etc.)
        # These get mapped to unused Unicode codepoints starting at 256
        n = 0
        for b in range(2**8):  # All 256 possible bytes
            if b not in bs:
                bs.append(b)
                cs.append(2**8 + n)  # Map to 256, 257, 258, ...
                n += 1

        # Convert codepoints to actual characters
        cs = [chr(n) for n in cs]

        # Return dict: byte → Unicode character
        return dict(zip(bs, cs))
    
    def pretokenize(self, text: str) -> List[str]:
        chunks = re.findall(self.pattern, text)
        return chunks
    
    def text_to_bytes(self, text: str) -> str:
        return "".join([self.byte_encoder[bt] for bt in text.encode("utf-8")])
    
    def bytes_to_text(self, byte_string: str) -> str:
        return bytes([self.byte_decoder[b] for b in byte_string]).decode('utf-8', errors='replace')
    
    def train(self, training_text: str, vocab_size: int = 1000):
        chunks = self.pretokenize(training_text)
        words = Counter() # initially contains tuple -> count. ('t', 'h', 'e') -> 1
        for chunk in chunks:
            byte_enc_chunk = self.text_to_bytes(chunk)
            word_tuple = tuple(byte_enc_chunk)
            words[word_tuple] += 1
        
        print(words)
        max_vocab_idx = len(self.vocab)
        num_merges = vocab_size - 256
        # now get the most frequent pair
        for i in range(num_merges):
            pair_counts = self._get_pair_counts(words)
            if len(pair_counts) == 0:
                print("We are done at ", i)
                break
            most_common_pair = pair_counts.most_common(1)[0] # will be a touple in format (wrd_seq, freq) => (('a', 's'), 3)
            best_pair = most_common_pair[0]
            # print("Best Pair ", most_common_pair)
            # print("WC bef: ", words)
            # merge the pair now
            words = self._merge_pair(words, best_pair)
            # print("WC After: ", words)
            # print("="*50, "\n")

            # record merges
            jp = "".join(best_pair)
            self.merges[best_pair] = jp
            self.merge_ranks[jp] = i
            self.vocab[jp] = max_vocab_idx

            max_vocab_idx += 1
        
        # build reverse vocab
        self.reverse_vocab = {tidx: t for t, tidx in self.vocab.items()}


    def _get_pair_counts(self, words_seq_counter: Dict[tuple, int]) -> Counter:
        pair_counts = Counter()
        for word_seq, freq in words_seq_counter.items():
            for i in range(len(word_seq)-1):
                pair = (word_seq[i], word_seq[i+1])
                pair_counts[pair] += freq
        
        return pair_counts
    
    def _merge_pair(self, words_seq_counter: Dict[str, int], pair: tuple) ->  Dict[str, int]:
        jp = "".join(pair)
        new_wrd_seqs = {}
        for word_seq, freq in words_seq_counter.items():
            new_wrd_seq = []
            i = 0
            while i < len(word_seq):
                if (i < len(word_seq) - 1) and (word_seq[i]+word_seq[i+1] == jp):
                    new_wrd_seq.append(jp)
                    i += 2
                else:
                    new_wrd_seq.append(word_seq[i])
                    i += 1
            
            new_wrd_seqs[tuple(new_wrd_seq)]  = freq
        
        return new_wrd_seqs
    
    def add_special_tokens(self, spl_tokens: list[str]):
        max_vocab_len = len(self.vocab)
        for spl_token in spl_tokens:
            self.vocab[spl_token] = max_vocab_len
            self.reverse_vocab[max_vocab_len] = spl_token
            self.special_tokens[spl_token] = max_vocab_len
            max_vocab_len += 1
        
        import re as std_re
        escaped_spl_tokens = [std_re.escape(st) for st in spl_tokens]
        self.special_tokens_pattern = re.compile("(" + "|".join(escaped_spl_tokens) + ")")
    
    def _split_on_spl_tokens(self, new_text):
        if not self.special_tokens_pattern:
            return [(new_text, False)]

        parts = self.special_tokens_pattern.split(new_text)
        result = []
        for p in parts:
            if not p: # empty string
                continue
            is_spl =  p in self.special_tokens
            result.append((p , is_spl))
        return result


    
    def encode(self, new_txt):
        if not new_txt:
            return []
        
        encoded_tokens = []
        
        parts_wid_spl_tokens = self._split_on_spl_tokens(new_text=new_txt)
        # print([j for j in parts_wid_spl_tokens])
        # print("parts_wid_spl_tokens ", len(parts_wid_spl_tokens))
        for part, is_spl in parts_wid_spl_tokens:
            if is_spl:
                encoded_tokens.append(part)
            else:
                chunks = self.pretokenize(part)
                
                for chunk in chunks:
                    byte_chunk = self.text_to_bytes(chunk)
                    tokens = self._apply_bpe(byte_chunk)
                    encoded_tokens.extend(tokens) # this will have word tokenks (safe byte represeted) ['Ġa', 's']
        # print("ECC ", encoded_tokens)
        return [self.vocab.get(t, -1) for t in encoded_tokens]
    
    def _apply_bpe(self, word_chunk):
        if len(word_chunk) == 1:
            return [word_chunk]

        if word_chunk in self.bpe_cache:
            return self.bpe_cache[word_chunk]
        
        if word_chunk in self.vocab:
            self.bpe_cache[word_chunk] = [word_chunk]
            return [word_chunk]
        
        split_wrd_chunk = list(word_chunk)

        while (True):
            # print("SPC ", split_wrd_chunk)
            best_pair = None
            best_merge_rank = float('inf')

            # get the best ranking for the pairs
            for i in range(len(split_wrd_chunk)-1):
                pair = split_wrd_chunk[i] + split_wrd_chunk[i+1]
                mr = self.merge_ranks.get(pair, float('inf'))
                if mr < best_merge_rank:
                    best_merge_rank = mr
                    best_pair = pair
                if best_merge_rank == 0:
                    best_pair = pair
                    break
            if float('inf') == best_merge_rank:
                break
            
            # now merge that best pair
            j = 0
            new_word_chunk = []
            while j < len(split_wrd_chunk):
                # print("NW = ", new_word_chunk, "j = ", j)
                if (j < len(split_wrd_chunk) -1) and (split_wrd_chunk[j]+split_wrd_chunk[j+1] == best_pair):
                    new_word_chunk.append(best_pair)
                    j += 2
                else:
                    new_word_chunk.append(split_wrd_chunk[j])
                    j += 1

            split_wrd_chunk = new_word_chunk
            if len(new_word_chunk) == 1:
                break

        self.bpe_cache[word_chunk] = split_wrd_chunk
        return split_wrd_chunk
    

    def decode(self, token_ids: List[int]): 
        token_bytes = [self.reverse_vocab.get(t, -1) for t in token_ids]
        byte_str = "".join(token_bytes)
        return self.bytes_to_text(byte_str)
        



if __name__ == "__main__":
    t = SBTokenizer()
    tx1 = "he is a good goon hence here referencing a fencing here తెలుగు భాగవతం"
    tx = "తెలుగు భాగవతం"
    # print(t.byte_decoder)
    t.train(tx1, vocab_size=300)
    special_tokens = [
    "<|endoftext|>",
    "<|startoftext|>",
    "<PAD>",
    "<UNK>",
    "<MASK>"
    ]
    t.add_special_tokens(spl_tokens=special_tokens)
    # c = t.encode("he <PAD> yes <UNK> goon Hood")
    # c1 = t.encode("వతం")

    # print(c)
    # print(c1)
    print("*****", t.encode("h"))
    print(t.decode([257, 220, 298, 220, 88, 68, 82, 220, 299, 277, 220, 39, 78, 78, 67]))
    print(t.decode([256, 113, 256, 97, 295]))

    # # for mt in t.merge_ranks:
    # #     print(mt , " => ", t.bytes_to_text(mt))
    # c = t.encode("భాగ<PAD> yes ")
    # print(c)
    # print(t.decode([256, 255, 265, 260, 272]))

