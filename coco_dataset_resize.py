import argparse
import json
import os
import cv2
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox
from collections import defaultdict

val_th = 20247
train_th = 24609

def resizeImageAndBoundingBoxes(imgFile, bboxes, inputW, inputH, targetImgW, targetImgH, outputImgFile):
    imgFile = imgFile.replace('jpg','png')
    split = imgFile.split("/")
    if ((split[2] == 'panoptic_val2017' and int(split[3].split(".")[0]) <= val_th) or (split[2] == 'panoptic_train2017' and int(split[3].split(".")[0]) <= train_th)):
        print("Reading image {0} ...".format(imgFile))
        img = cv2.imread(imgFile)
        if inputW > inputH:
            seq = iaa.Sequential([
                                    iaa.Resize({"height": "keep-aspect-ratio", "width": targetImgW}),
                                    iaa.PadToFixedSize(width=targetImgW, height=targetImgH)
                                ])
        else:
            seq = iaa.Sequential([
                                    iaa.Resize({"height": targetImgH, "width": "keep-aspect-ratio"}),
                                    iaa.PadToFixedSize(width=targetImgW, height=targetImgH)
                                ])
        
        image_aug, bbs_aug = seq(image=img, bounding_boxes=bboxes)

        print("Writing resized image {0} ...".format(outputImgFile))
        cv2.imwrite(outputImgFile, image_aug)
        print("Resized image {0} written successfully.".format(outputImgFile))

        return bbs_aug
    
    return None


if __name__ == "__main__":

    ia.seed(1)

    ap = argparse.ArgumentParser()

    ap.add_argument("-i", "--images_dir", required=True, help="Directory where are located the images referenced in the annotations file")
    ap.add_argument("-a", "--annotations_file", required=True, help="COCO JSON format annotations file")
    ap.add_argument("-w", "--image_width", required=True, help="Target image width")
    ap.add_argument("-t", "--image_height", required=True, help="Target image height")
    ap.add_argument("-o", "--output_ann_file", required=True, help="Output annotations file")
    ap.add_argument("-f", "--output_img_dir", required=True, help="Output images directory")

    args = vars(ap.parse_args())

    imageDir                = args['images_dir']
    annotationsFile         = args['annotations_file']
    targetImgW              = int(args['image_width'])
    targetImgH              = int(args['image_height'])
    outputImageDir          = args['output_img_dir']
    outputAnnotationsFile   = args['output_ann_file']

    print("Loading annotations file...")
    data = json.load(open(annotationsFile, 'r'))
    print("Annotations file loaded.")

    print("Building dictionnaries...")
    anns    = defaultdict(list)
    annsIdx = dict()
    for i in range(0, len(data['annotations'])):
        anns[data['annotations'][i]['image_id']].append(data['annotations'][i])
        segments_info = data['annotations'][i]['segments_info']
        for j in range(0, len(segments_info)):
            annsIdx[segments_info[j]['id']] = i
    print("Dictionnaries built.")
    
    for img in data['images']:
        print("Processing image file {0} and its bounding boxes...".format(img['file_name']))
        annList = anns[img['id']][0]['segments_info']
        bboxesList = []
        for ann in annList:
            bboxData = ann['bbox']
            bboxesList.append(BoundingBox(x1=bboxData[0], y1=bboxData[1], x2=bboxData[0] + bboxData[2], y2=bboxData[1] + bboxData[3]))

        imgFullPath         = os.path.join(imageDir, img['file_name'])
        file_name = img['file_name'].replace('jpg','png')
        outputImgFullPath   = os.path.join(outputImageDir, file_name)
        outputDir           = os.path.dirname(outputImgFullPath)

        if not os.path.exists(outputDir):
            os.makedirs(outputDir)
        
        outNewBBoxes = resizeImageAndBoundingBoxes(imgFullPath, bboxesList, int(img['width']), int(img['height']), targetImgW, targetImgH, outputImgFullPath)
        if outNewBBoxes is not None:
            for i in range(0, len(annList)):
                annId = annList[i]['id']
                segments_info = data['annotations'][annsIdx[annId]]['segments_info'][0]
                segments_info[0] = outNewBBoxes[i].x1
                segments_info[1] = outNewBBoxes[i].y1
                segments_info[2] = outNewBBoxes[i].x2 - outNewBBoxes[i].x1
                segments_info[3] = outNewBBoxes[i].y2 - outNewBBoxes[i].y1
            
        img['width']    = targetImgW
        img['height']   = targetImgH
    
    print("Writing modified annotations to file...")
    with open(outputAnnotationsFile, 'w') as outfile:
        json.dump(data, outfile)
    
    print("Finished.")