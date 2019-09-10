import skimage.io as io
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

data_dir = "/home/tcai/Documents/nlp/final_project"
data_type = "train2017"
annotation_file = data_dir + '/annotations/captions_%s.json' % data_type
image_dir = data_dir + '/%s/' % data_type
coco_caps = COCO(annotation_file)

# Obtain categories
annFile = '{}/annotations/instances_{}.json'.format(data_dir, data_type)
coco = COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]
print('\nCOCO Categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO Supercategories: \n{}'.format(' '.join(nms)))

# Show sample data set by choosing categories
temp_cat = ['dog', 'person', 'ball']

catIds = coco.getCatIds(catNms = temp_cat)
imgIds = coco.getImgIds(catIds = catIds)

# Identify relevant images
if len(imgIds) > 0:
    imgIds = coco.getImgIds(imgIds = imgIds[-1])  # Pick the last image
    print("The index of the chosen image is %s.\n" % (str(imgIds[0])))
else:
    print("No matched images found.")

# Load and display captions
annIds = coco_caps.getAnnIds(imgIds)
anns = coco_caps.loadAnns(annIds)
print("The corresponding captions are:")
coco_caps.showAnns(anns)

# Show image
img = coco.loadImgs(imgIds)[0]
I = io.imread('%s/%s/%s' % (data_dir, data_type, img['file_name']))
plt.imshow(I)
plt.axis('off')
plt.show()

"""Deprecated code"""
# # Similar function for testing how inception v3 model works
# def inceptionv3_predict(image_path, image_model):
#     # Preprocess images
#     img = tf.keras.preprocessing.image.load_img(image_path, target_size = (299, 299))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis = 0)
#     img_array = keras.applications.inception_v3.preprocess_input(img_array)
#     preds = image_model.predict(img_array)
#     P = keras.applications.imagenet_utils.decode_predictions(preds)
#
#     # Show prediction result
#     for (i, (imagenetID, label, prob)) in enumerate(P[0]):
#         print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))
#
#     # Show image
#     I = io.imread(image_path)
#     plt.imshow(I)
#     plt.axis('off')
#     plt.show()
#
# # Randomly choose images to be predicted by Inception V3
# random_img_id = np.random.choice(all_img_name_vector)
# random_img_path = all_img_path_vector[random_img_id]
# inceptionv3_predict(random_img_path, image_model)