import json
from nltk.tokenize import sent_tokenize

input_file = 'transcripts_varied.json'
output_file = 'transcripts_varied.jsonl'
jsonl_dicts = []



if __name__ == "__main__":
    with open(input_file, encoding='utf8') as f:
        data = json.loads(f.read())


    for episode_name, text in data.items():
        sentences = sent_tokenize(text)
        for sentence_id, sent in enumerate(sentences):
            if sent == '' or sent == ' ': continue
            if sent[0] == ' ': sent = sent[1:]

            jsonl_dict = {'ep_id': episode_name, 'sent_id': str(sentence_id),
                          'text': str(sent).encode('ascii', 'ignore').decode('ascii').strip('\n')}
            jsonl_dicts.append(jsonl_dict)

with open(output_file, 'w') as f:
    for d in jsonl_dicts:
        json.dump(d, f)
        f.write('\n')
