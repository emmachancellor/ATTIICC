from attiicc.segmentation.sam import SamSegmenter

image_path = "/media/ss4tbSSD/EDYER/nanowell_processing/images/p00_0_A01f00d0_2.png"

test_sam = SamSegmenter("/media/ss4tbSSD/EDYER/nanowell_processing/weights/sam_vit_h_4b8939.pth", image_path=image_path)
#test_sam.plot_segmented_image(save=True, save_path="/media/ss4tbSSD/EDYER/nanowell_processing/test_img.png")
#test_sam.plot_masks(save=True, save_path="/media/ss4tbSSD/EDYER/nanowell_processing/test_masks.png")

print("Image Area: ", test_sam.area)