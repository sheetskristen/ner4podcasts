from typing import Mapping, Sequence, Dict, Optional

from spacy.language import Language
from spacy.tokens import Doc

from hw3utils import PRF1
import json
import random
import copy
import pickle
import os
from collections import Counter


from typing import Sequence, Dict, Optional, List

from spacy.tokens import Doc
import pymagnitude
import csv
from typing import Sequence, Dict, List, NamedTuple, Tuple, Counter
from hw3utils import FeatureExtractor, ScoringCounts, ScoringEntity
from spacy.tokens import Span
from typing import Iterable, Sequence, Tuple, List, Dict

from nltk import ConfusionMatrix
from spacy.tokens import Span, Doc, Token

from hw3utils import FeatureExtractor, EntityEncoder, PRF1

from hw3utils import PUNC_REPEAT_RE, DIGIT_RE, UPPERCASE_RE, LOWERCASE_RE


from spacy.tokens import Token

import pycrfsuite

from collections import defaultdict

import sys
from decimal import ROUND_HALF_UP, Context

import spacy


def ingest_json_document(doc_json: Mapping, nlp: Language) -> Doc:
    #json_as_dict = json.loads(doc_json)
    if not doc_json["annotation_approver"] and not doc_json["labels"]:
        raise ValueError("Instance is not annotated!")
    else:
        doc = nlp(doc_json["text"])
        #print(doc)
        #print(doc_json["id"])
        spans = list()
        for label in doc_json["labels"]:
            start_char =  label[0]
            end_char = label[1]
            tag = label[2]
            token_start = get_starting_token(start_char, doc)
            token_end = get_ending_token(end_char, doc)
            if token_start is None or token_end is None:
                raise ValueError("Token alignment impossible!")
            spans.append(Span(doc, token_start, token_end, tag))
        doc.ents = spans
        return doc
    #except ValueError:
        #print(("Instance is not annotated!"))


def get_starting_token(start_char, doc):
    for token in doc:
        if start_char <= token.idx:
            return token.i
    return None

def get_ending_token(end_char, doc):
    for token in doc:
        if end_char <= token.idx:
            return token.i
    return None

def span_prf1_type_map(
    reference_docs: Sequence[Doc],
    test_docs: Sequence[Doc],
    type_map: Optional[Mapping[str, str]] = None,
) -> Dict[str, PRF1]:
    """
    if type_map is not None:
        remapping(reference_docs, type_map) # ugly code, but otherwise the
        remapping(test_docs, type_map)
    """
    counts = defaultdict(lambda: defaultdict(int))
    for i in range(len(reference_docs)):
        ents_from_ref = {ent for ent in reference_docs[i].ents}
        ents_from_test = {ent for ent in test_docs[i].ents}
        if type_map is not None:
            ents_from_ref = remapping(ents_from_ref, type_map)  # ugly code, but otherwise the
            ents_from_test = remapping(ents_from_test, type_map)
        for ent_test in ents_from_test:
            if is_ent_in_list(ent_test, ents_from_ref):
                counts[ent_test.label_]["tp"] += 1
            else:
                counts[ent_test.label_]["fp"] += 1
        for ent_ref in ents_from_ref:
            if not is_ent_in_list(ent_ref, ents_from_test):
                counts[ent_ref.label_]["fn"] += 1
    prf1 = dict()
    for key, value in counts.items():
        precision = get_precision(counts[key]["tp"], counts[key]["fp"])
        recall = get_recall(counts[key]["tp"], counts[key]["fn"])
        f1 = get_f1(precision, recall)
        prf1[key] = PRF1(precision, recall, f1)
    get_prf1_all(counts, prf1)
    #print(counts)
    return prf1

def remapping(ents, type_map):
    new_ents = set()
    for ent in ents:
        if ent.label_ in type_map.keys():
            new_ents.add(Span(ent.doc, ent.start, ent.end, type_map[ent.label_]))
        else:
            new_ents.add(ent)
    return new_ents


def is_ent_in_list(ent_ref, list):
    for elem in list:
        if same_ents(ent_ref, elem):
            return True
    return False

def get_prf1_all(counts, prf1):
    tp_all = 0
    fp_all = 0
    fn_all = 0
    for ent, values in counts.items():
        tp_all += counts[ent]["tp"]
        fp_all += counts[ent]["fp"]
        fn_all += counts[ent]["fn"]
    precision_all = get_precision(tp_all, fp_all)
    recall_all = get_recall(tp_all, fn_all)
    prf1[""] = PRF1(precision_all, recall_all, get_f1(precision_all, recall_all))

def same_ents(ent1, ent2):
    return ent1.label_ == ent2.label_ and ent1.start == ent2.start and ent1.end == ent2.end

def get_precision(tp, fp):
    if tp + fp == 0:
        return 0
    return tp/(tp+fp)

def get_recall(tp, fn):
    if tp + fn == 0:
        return 0
    return tp/(tp+fn)

def get_f1(precision, recall):
    if precision + recall == 0:
        return 0
    return 2*precision*recall/(precision+recall)


class WindowedTokenFeatureExtractor:
    def __init__(self, feature_extractors: Sequence[FeatureExtractor], window_size: int):
        self.extractors = feature_extractors
        self.window_size = window_size

    def extract(self, tokens: Sequence[str]) -> List[Dict[str, float]]:
        featurized = []
        for i in range(0, len(tokens)):
            dict_feat = dict()
            token = tokens[i]
            for extractor in self.extractors:
                extractor.extract(token, i, 0, tokens, dict_feat)
                for j in range(1, self.window_size + 1):
                    if i - j >= 0:
                        extractor.extract(tokens[i - j], i - j, -j, tokens, dict_feat)
                    if i + j < len(tokens):
                        extractor.extract(tokens[i + j], i + j, j, tokens, dict_feat)
            featurized.append(dict_feat)
        return featurized

class CRFsuiteEntityRecognizer:
    def __init__(
        self, feature_extractor: WindowedTokenFeatureExtractor, encoder: EntityEncoder
    ) -> None:
        self.feature_extractor = feature_extractor
        self._encoder = encoder

    @property
    def encoder(self) -> EntityEncoder:
        return self._encoder

    def set_encoder(self, encoder):
        self._encoder = encoder

    def train(self, docs: Iterable[Doc], algorithm: str, params: dict, path: str) -> None:
        trainer = pycrfsuite.Trainer(algorithm, verbose=False)
        trainer.set_params(params)
        for doc in docs:
            #print(doc)
            for sent in doc.sents:
                tokens = list(sent)
                features = self.feature_extractor.extract([token.text for token in tokens])
                encoded_labels = self._encoder.encode(tokens)
                trainer.append(features, encoded_labels)
        trainer.train(path)
        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(path)

    def __call__(self, doc: Doc) -> Doc:
        if not self.tagger:
            raise ValueError('train() method should be called first!')
        entities = list()
        #print(doc.ents)
        for sent in doc.sents:
            tokens = list(sent)
            tags = self.predict_labels(tokens)
            entities.append(decode_bilou(tags, tokens, doc))
        doc.ents = [item for sublist in entities for item in sublist]
        #print(doc.ents)
        return doc

    def predict_labels(self, tokens: Sequence[str]) -> List[str]:
        features = self.feature_extractor.extract([str(token) for token in tokens])
        tags = self.tagger.tag(features)
        return tags

class BILOUEncoder(EntityEncoder):
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        encoded = []
        labels = [token.ent_iob_ + "-" + token.ent_type_ for token in tokens]
        spans = decode_bilou(labels, tokens, tokens[0].doc)
        for span in spans:
            if self.is_unitary(span):
                encoded.append("U-" + span.label_)
            elif self.is_empty(span):
                for i in range(span.start, span.end):
                    encoded.append("O")
            else:
                encoded.append("B-" + span.label_)
                for i in range(span.start + 1, span.end-1):
                    encoded.append("I-" + span.label_)
                encoded.append("L-" + span.label_)
        return encoded

    def is_unitary(self, span):
        return span.end == span.start + 1 and span.label > 0

    def is_empty(self, span):
        return span.label == 0


class BIOEncoder(EntityEncoder):
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        labels = [token.ent_iob_ + "-" + token.ent_type_ for token in tokens]
        spans = decode_bilou(labels, tokens, tokens[0].doc)
        encoded = []
        for span in spans:
            if self.is_empty(span):
                for i in range(span.start, span.end):
                    encoded.append("O")
            else:
                encoded.append("B-" + span.label_)
                for i in range(span.start + 1, span.end):
                    encoded.append("I-" + span.label_)
        return encoded

    def is_empty(self, span):
        return span.label == 0


class IOEncoder(EntityEncoder):
    def encode(self, tokens: Sequence[Token]) -> List[str]:
        labels = [token.ent_iob_ + "-" + token.ent_type_ for token in tokens]
        spans = decode_bilou(labels, tokens, tokens[0].doc)
        encoded = []
        for span in spans:
            if self.is_empty(span):
                for i in range(span.start, span.end):
                    encoded.append("O")
            else:
                for i in range(span.start, span.end):
                    encoded.append("I-" + span.label_)
        return encoded

    def is_empty(self, span):
        return span.label == 0

def decode_bilou(labels: Sequence[str], tokens: Sequence[Token], doc: Doc) -> List[Span]:
    spans = []
    tag_interruptus = False
    span_type = None
    i = 0
    initial = None
    while i < len(labels):
        label = labels[i]
        if (label == "O"):  # The current label is O
            if tag_interruptus:  # If we were in the middle of a span, we create it and append it
                spans.append(Span(doc, tokens[initial].i, tokens[i].i, span_type))
            tag_interruptus = False  # We are no longer in the middle of a tag
        else:  # The current label is B I L or U
            label_type = label.split("-")[1]  # We get the type (PER, MISC, etc)
            if tag_interruptus:  # if we were in the middle of a tag
                if label_type != span_type:  # and the types dont match
                    spans.append(Span(doc, tokens[initial].i, tokens[i].i,
                                      span_type))  # we close the previous span and append it
                    span_type = label_type  # we are now in the middle of a new span
                    initial = i
            else:  # we were not in the middle of a span
                initial = i  # initial position will be the current position
                tag_interruptus = True  # we are now in the middle of a span
                span_type = label_type
        i = i + 1
    if tag_interruptus:  # this covers entities at the end of the sentence (we left a tag_interruptus at the end of the list)
        spans.append(Span(doc, tokens[initial].i, tokens[i-1].i + 1, span_type))
    return spans

class BiasFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str], features: Dict[str, float]):
        if relative_idx == 0:
            features["bias"] = 1.0

class TokenFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str], features: Dict[str, float]):
        features["tok["+str(relative_idx)+"]="+token]=1.0

class PrefixFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str], features: Dict[str, float]):
        features["tok["+str(relative_idx)+"]="+token[:4]]=1.0

class SuffixFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str], features: Dict[str, float]):
        features["tok["+str(relative_idx)+"]="+token[4:]]=1.0

class UppercaseFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str], features: Dict[str, float]):
        if token.isupper():
            features["uppercase["+str(relative_idx)+"]"]=1.0

class HasApostrophe(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str], features: Dict[str, float]):
        if token.endswith("'s"):
            features["has_apostrophe["+str(relative_idx)+"]"]=1.0

class IsShortWord(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str], features: Dict[str, float]):
        if len(token)<4:
            features["is_short["+str(relative_idx)+"]"]=1.0

class IsQualWord(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str], features: Dict[str, float]):
        quality_words = ["good", "bad", "terrible", "awful", "nice", "lovely", "great"]
        if token in quality_words:
            features["quality_word["+str(relative_idx)+"]"]=1.0

class WordEnding(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str], features: Dict[str, float]):
        features["ending["+str(relative_idx)+"]="+token[-3:]]=1.0


class TitlecaseFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str], features: Dict[str, float]):
        if token.istitle():
            features["titlecase["+str(relative_idx)+"]"]=1.0


class InitialTitlecaseFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str], features: Dict[str, float]):
        if token.istitle() and current_idx == 0:
            features["initialtitlecase["+str(relative_idx)+"]"] = 1.0


class PunctuationFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str], features: Dict[str, float]):
        if PUNC_REPEAT_RE.match(token):
            features["punc["+str(relative_idx)+"]"] = 1.0


class DigitFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str], features: Dict[str, float]):
        if DIGIT_RE.search(token):
            features["digit["+str(relative_idx)+"]"] = 1.0


class WordShapeFeature(FeatureExtractor):
    def extract(self, token: str, current_idx: int, relative_idx: int, tokens: Sequence[str], features: Dict[str, float]):
        shape = []
        for letter in token:
            if DIGIT_RE.search(letter):
                shape.append("0")
            elif LOWERCASE_RE.search(letter):
                shape.append('x')
            elif UPPERCASE_RE.search(letter):
                shape.append('X')
            else:
                shape.append(letter)
        features["shape["+str(relative_idx)+"]="+("".join(shape))] = 1.0

class WordVectorFeature(FeatureExtractor):
    def __init__(self, vectors_path: str, scaling: float = 1.0) -> None:
        self.vectors = pymagnitude.Magnitude(vectors_path, normalized=False)
        self.scaling = scaling

    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if relative_idx  == 0:
            word_vector = self.vectors.query(token)
            keys = self.get_keys(word_vector)
            features.update(zip(keys, self.scaling*word_vector))

    def get_keys(self, word_vector):
        return ["v"+str(i) for i in range(len(word_vector))]



class BrownClusterFeature(FeatureExtractor):
    def __init__(
        self,
        clusters_path: str,
        *,
        use_full_paths: bool = False,
        use_prefixes: bool = False,
        prefixes: Optional[Sequence[int]] = None,
    ):
       if not use_full_paths and not use_prefixes:
           raise ValueError('Either use_full_paths or use_prefixes has to be True!')
       self.use_full_paths = use_full_paths
       self.use_prefixes = use_prefixes
       self.prefixes = prefixes
       self.clusters = dict()
       self.populate_clusters(clusters_path)

    def populate_clusters(self, path):
       with open(path, encoding="utf-8") as tsv:
           for line in csv.reader(tsv, delimiter="\t", quoting=csv.QUOTE_NONE):
               self.clusters[line[1]] = line[0]


    def extract(
        self,
        token: str,
        current_idx: int,
        relative_idx: int,
        tokens: Sequence[str],
        features: Dict[str, float],
    ) -> None:
        if relative_idx == 0 and token in self.clusters:
            if self.use_full_paths:
                features["cpath="+self.clusters[token]] = 1.0

            if self.use_prefixes:
                path = self.clusters[token]
                if not self.prefixes:
                    for i in range(1, len(path)+1):
                        features["cprefix"+str(i)+"="+path[:i]] = 1.0
                else:
                    for prefix in self.prefixes:
                        if prefix <= len(path):
                            features["cprefix" + str(prefix) + "=" + path[:prefix]] = 1.0







def span_scoring_counts(
    reference_docs: Sequence[Doc], test_docs: Sequence[Doc], typed: bool = True
) -> ScoringCounts:
    tp = []
    fp = []
    fn = []
    for i in range(len(reference_docs)):
        doc_reference = reference_docs[i]
        doc_test = test_docs[i]
        #if not typed:
        #remove_labels(doc_reference, doc_test)
        for ent_test in doc_test.ents:
            if is_ent_in_list(ent_test, doc_reference.ents):
                tp.append(ScoringEntity(tuple(ent_test.text.split()), ent_test.label_))
            else:
                fp.append(ScoringEntity(tuple(ent_test.text.split()), ent_test.label_))
        for ent_ref in doc_reference.ents:
            if not is_ent_in_list(ent_ref, doc_test.ents):
                fn.append(ScoringEntity(tuple(ent_ref.text.split()), ent_ref.label_))
    return ScoringCounts(Counter(tp), Counter(fp), Counter(fn))


def get_metrics(scores):
    metrics = [get_precision(scores), get_recall(scores), get_f1(scores)]
    rounder = Context(rounding=ROUND_HALF_UP, prec=4)
    metrics_string = [str(rounder.create_decimal_from_float(num * 100)) for num in metrics]
    return ("\t".join(metrics_string))


def generate_tsvs(ep_ids, predicted):
    with open('topic_modeling_tsvs/predicted_ents.tsv', 'w') as f:
        f.write('entity\ttype\ttopic\n')
        for ep_id, pred in zip(ep_ids, predicted):
            ents = pred.ents
            for ent in ents:
                f.write(f'{ent}\t{ent.label_}\t{ep_id}\n')


def main() -> None:
    NLP = spacy.load("en_core_web_sm", disable=["ner"])
    docs = list()
    path_to_annotated = "annotated_data"
    for dirpath, dirnames, filenames in os.walk(path_to_annotated):
        for file in filenames:
            with open(path_to_annotated+"/"+file, "r", encoding="utf8") as f:
                lines = f.readlines()
            for line in lines:
                json_as_dict = json.loads(line)
                try:
                    doc = ingest_json_document(json_as_dict, NLP)
                    docs.append((doc, json_as_dict['ep_id']))
                except ValueError:
                    pass

    random.shuffle(docs)
    """
    I chose to pickle in order to: 
    - avoid doing everything from the begining every time I run the code
    - make sure that the experiements are run on the same test/train split (so that metrics are comparable)

    with open('data.pickle', 'wb') as handle:
        pickle.dump(docs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('data.pickle', 'rb') as handle:
        docs = pickle.load(handle)
    """

    docs, ep_ids = zip(*docs)

    corpus_description(docs)
    gold = docs[:len(docs)//5]
    predicted = copy.deepcopy(gold)
    for doc in predicted:
        doc.ents = []
    training = docs[len(docs)//5:]
    # best configuration features
    features = [BiasFeature(), TokenFeature(), UppercaseFeature(), TitlecaseFeature(),
                DigitFeature(), WordShapeFeature(), WordVectorFeature(("wordvectors/wiki-news-300d-1M-subword.magnitude"), scaling=2.0),
                BrownClusterFeature("wordvectors/rcv1.64M-c10240-p1.paths", use_full_paths=True), SuffixFeature(), PrefixFeature()]

    #PunctuationFeature(),

    crf = CRFsuiteEntityRecognizer(WindowedTokenFeatureExtractor(features,1,),BILOUEncoder())
    crf.train(training, "ap", {"max_iterations":  40}, "tmp.model")

    predicted = [crf(doc) for doc in predicted]

    # This generates TSVs for topic modeling.
    generate_tsvs(ep_ids, predicted)

    prf1 = span_prf1_type_map(gold, predicted, , {"LOCATION":"GPE_LOC", "GPE":"GPE_LOC"})

    print_results(prf1)
    print(span_scoring_counts(gold, predicted))

def corpus_description(docs):
    instances = len(docs)
    entities = [ent.label_ for doc in docs for ent in doc.ents]
    entities_total = sum([len(doc.ents) for doc in docs])
    tokens = sum([len(doc) for doc in docs])
    print("Instances: " + str(instances))
    print("Entities: " + str(entities_total))
    print("Tokens: " + str(tokens))
    print(Counter(entities))


def print_results(prf1):
    # Always round .5 up, not towards even numbers as is the default
    rounder = Context(rounding=ROUND_HALF_UP, prec=4)
    print("{:30s} {:30s}".format("Tag", "Prec\t\tRec\t\tF1"), file=sys.stderr)
    for ent_type, score in sorted(prf1.items()):
        if ent_type == "":
            ent_type = "ALL"
        """
        fields = [ent_type] + [
            str(rounder.create_decimal_from_float(num * 100)) for num in score
        ]
        """
        metrics = [str(float(rounder.create_decimal_from_float(num * 100))) for num in score]
        print("{:30s} {:30s}".format(ent_type, "\t".join(metrics)), file=sys.stderr)


if __name__ == "__main__":
    main()
