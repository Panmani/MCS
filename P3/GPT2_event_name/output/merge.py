import json
import os

merged_vcr_high_caps = []
for part_idx in range(8):
    with open(os.path.join("vcr_part{}_from_8parts_all_sample_1_num_5_top_k_0_top_p_0.9.json".format(part_idx) ), "r") as output_file:
        vcr_high_caps = json.load(output_file)
        merged_vcr_high_caps += vcr_high_caps
    # print(type(vcr_high_caps)) # list
    # merged_vcr_captions.update(vcr_captions)

print(len(merged_vcr_high_caps))

# vcr_records = [] # This format is specifically for GPT2
# for img_fn in merged_vcr_captions:
#     for cap in merged_vcr_captions[img_fn][:top_k]:
#         # output_keys = ['img_fn', 'movie', 'metadata_fn', 'split', 'event', 'inference_relation', 'event_idx']
#         record = {
#             "img_fn"      : img_fn,
#             "movie"       : img_fn.split("/")[0],
#             "metadata_fn" : img_fn[:-len(".jpg")] + ".json",
#             "split"       : "all",
#             "event"       : cap,
#         }
#         vcr_records.append(record)
# print("Image Event Pairs:", len(vcr_records))

with open("vcr_high_caps_all.json", "w") as output_file:
    json.dump(merged_vcr_high_caps, output_file)
