import argparse
import json

# This file takes an annotated jsonl file and generates a jsonl file where each json dict contains a field 'ep_id' that indicates which podcast
# the transcript belongs to. This is necessary for topic modeling.

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', help='input jsonl file that requires ep_ids')
    parser.add_argument('--ep_id', help='whatever should go in the ep_id field')
    args = parser.parse_args()

    input_file = args.in_file
    ep_id = args.ep_id

    output_file = input_file[:-6] + '_with_ep_ids.jsonl'

    with open(input_file, 'r') as f:
        lines = f.read().split('\n')
        dicts = [json.loads(line) for line in lines if line != '']

        for d in dicts:
            d['ep_id'] = ep_id
    
    with open(output_file, 'w') as f:
        for d in dicts:
            f.write(json.dumps(d))
            f.write('\n')
