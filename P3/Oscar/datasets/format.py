import json

ANN_FILE = 'train_caption.json'
JSON_FILE = "train_caption_coco_format.json"

with open(ANN_FILE, "r") as ann_file:
    ann_list = json.load(ann_file)

images = []
for ann in ann_list:
    cur_image = {'id'        : ann['image_id'],
                 'file_name' : ann['image_id'],}
    images.append(cur_image)

format_json = {"annotations" : ann_list,
               "images"      : images,
               "type"        : "captions",
               "info"        : "dummy",
               "licenses"    : "dummy"}

with open(JSON_FILE, "w") as format_file:
    json.dump(format_json, format_file)

# with open(JSON_FILE, "r") as format_file:
#     format_json = json.load(format_file)
#
# for key in format_json.keys():
#     print(key)
#
# # annotations
# # images
# # type
# # info
# # licenses
#
# print(format_json['images'])
#
# all_image_id = []
# for entry in format_json['annotations']:
#     all_image_id.append(entry['image_id'])
#
# for entry in format_json['images']:
#     if entry['id'] != entry['file_name']:
#         print(entry['id'], entry['file_name'])
#
#     if entry['id'] not in all_image_id:
#         print(entry['id'])
