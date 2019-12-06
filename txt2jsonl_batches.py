import json
from nltk.tokenize import sent_tokenize

input_file = 'unannotated_data/transcripts/chapo.txt'
output_file = 'chapo.jsonl'
episode_name = "225"
jsonl_dicts = []


if __name__ == "__main__":
    batch = 1
    with open(input_file, encoding='utf8') as f:
        data = f.readlines()
        # print(data)

    for text in data:
        print(text)
        sentences = sent_tokenize(text)
        batch_dict = []
        for sentence_id, sent in enumerate(sentences):
            if sent == '' or sent == ' ': continue
            if sent[0] == ' ': sent = sent[1:]
            #
            jsonl_dict = {'ep_id': episode_name, 'sent_id': str(sentence_id),
                          'text': str(sent).encode('ascii', 'ignore').decode('ascii').strip('\n')}
            batch_dict.append(jsonl_dict)

        if batch_dict:
            batch_file = str(batch) + "_" + output_file
            with open(batch_file, 'w') as f:
                for d in batch_dict:
                    json.dump(d, f)
                    f.write('\n')
                batch += 1