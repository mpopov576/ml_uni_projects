from pycocotools.coco import COCO
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

coco = COCO("annotations/instances_train2017.json")

#print("Images:", len(coco.imgs))
#print("Categories:", len(coco.cats))
#print("Annotations:", len(coco.anns))

cat_ids = coco.getCatIds()
cats = coco.loadCats(cat_ids)

cat_names = {c["id"]: c["name"] for c in cats}

ann_ids = coco.getAnnIds()
anns = coco.loadAnns(ann_ids)

counter = Counter([ann["category_id"] for ann in anns])

labels = [cat_names[k] for k in counter.keys()]
values = list(counter.values())

plt.figure(figsize=(12,5))
plt.bar(labels[:10], values[:10])
plt.xticks(rotation=45)
plt.title("Top COCO Classes")
plt.show()

#############

img_ids = coco.getImgIds()

objects_per_image = []

for img_id in img_ids[:5000]:
    ann_ids = coco.getAnnIds(imgIds=[img_id])
    objects_per_image.append(len(ann_ids))

plt.figure()
sns.histplot(objects_per_image, bins=20)
plt.title("Objects per Image")
plt.show()

print("Avg objects per image:", sum(objects_per_image)/len(objects_per_image))

#############

widths = []
heights = []

for ann_id in list(coco.anns.keys())[:50000]:
    ann = coco.anns[ann_id]
    w, h = ann["bbox"][2], ann["bbox"][3]
    widths.append(w)
    heights.append(h)

plt.figure()
plt.scatter(widths[:5000], heights[:5000], alpha=0.3)
plt.title("Bounding Box Width vs Height")
plt.xlabel("Width")
plt.ylabel("Height")
plt.show()

######

print("Min width:", min(widths))
print("Max width:", max(widths))
print("Min height:", min(heights))
print("Max height:", max(heights))
