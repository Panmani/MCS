import json
import os

top_k = 1

merged_vcr_captions = {}
for split_idx in range(16):
    with open(os.path.join("output", "vcr_captions_{}.json".format(split_idx) ), "r") as output_file:
        vcr_captions = json.load(output_file)
    print(len(vcr_captions))
    merged_vcr_captions.update(vcr_captions)

print(len(merged_vcr_captions))

vcr_records = [] # This format is specifically for GPT2
for img_fn in merged_vcr_captions:
    for cap in merged_vcr_captions[img_fn][:top_k]:
        # output_keys = ['img_fn', 'movie', 'metadata_fn', 'split', 'event', 'inference_relation', 'event_idx']
        record = {
            "img_fn"      : img_fn,
            "movie"       : img_fn.split("/")[0],
            "metadata_fn" : img_fn[:-len(".jpg")] + ".json",
            "split"       : "all",
            "event"       : cap,
        }
        vcr_records.append(record)
print("Image Event Pairs:", len(vcr_records))

with open("vcr_captions_all.json", "w") as output_file:
    json.dump(vcr_records, output_file)
