import ccov
import cv2
import os
import glob
import shutil

# Execute the algorithm for all the IMG images in the test set
curr_dir_path = os.path.dirname(os.path.realpath(__file__))
output_dir = curr_dir_path + "/output/"
files = glob.glob(curr_dir_path + '/IMG/**/*.JPG', recursive=True)

shutil.rmtree(output_dir, ignore_errors=True)
os.makedirs(output_dir)

for file in files:
    print(os.path.basename(file))
    out_img = ccov.carcass_coverage(file)
    cv2.imwrite(output_dir + os.path.basename(file), out_img)


# IMPORTANT CASES - Testing purposes

# cv2.imshow("img", ccov.carcass_coverage("AOL4/46/DSC01363.JPG"))
# cv2.imshow("img", ccov.carcass_coverage("AOL4/46/DSC01362.JPG"))
# cv2.imshow("img", ccov.carcass_coverage("AOL4/50/CIMG2561.JPG"))
# cv2.imshow("img", ccov.carcass_coverage("AOL1/04/DSC01428.JPG"))
# cv2.imshow("img", ccov.carcass_coverage("AOL3/17/DSC01340.JPG"))
# cv2.imshow("img", ccov.carcass_coverage("AOL2/45/DSC01252.JPG"))
# cv2.imshow("img", ccov.carcass_coverage("AOL3/31/DSC01416.JPG"))
# cv2.imshow("img", ccov.carcass_coverage("AOL2/47/DSC01279.JPG"))
# cv2.imshow("img", ccov.carcass_coverage("AOL4/59/DSC01360.JPG"))
# cv2.imshow("img", ccov.carcass_coverage("AOL2/48/DSC01277.JPG"))
cv2.waitKey(0)
