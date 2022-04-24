# The contents of this file are in the public domain.
#
#   This example program shows how to find frontal human faces in an image and
#   estimate their pose.  The pose takes the form of 68 landmarks.  These are
#   points on the face such as the corners of the mouth, along the eyebrows, on
#   the eyes, and so forth.
#
#   The face detector we use is made using the classic Histogram of Oriented
#   Gradients (HOG) feature combined with a linear classifier, an image pyramid,
#   and sliding window detection scheme.  The pose estimator was created by
#   using dlib's implementation of the paper:
#      One Millisecond Face Alignment with an Ensemble of Regression Trees by
#      Vahid Kazemi and Josephine Sullivan, CVPR 2014
#   and was trained on the iBUG 300-W face landmark dataset (see
#   https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/):  
#      C. Sagonas, E. Antonakos, G, Tzimiropoulos, S. Zafeiriou, M. Pantic. 
#      300 faces In-the-wild challenge: Database and results. 
#      Image and Vision Computing (IMAVIS), Special Issue on Facial Landmark Localisation "In-The-Wild". 2016.
#   You can get the trained model file from:
#   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2.
#   Note that the license for the iBUG 300-W dataset excludes commercial use.
#   So you should contact Imperial College London to find out if it's OK for
#   you to use this model file in a commercial product.
#
#
#   Also, note that you can train your own models using dlib's machine learning
#   tools. See train_shape_predictor.py to see an example.
#
#
# COMPILING/INSTALLING THE DLIB PYTHON INTERFACE
#   You can install dlib using the command:
#       pip install dlib
#
#   Alternatively, if you want to compile dlib yourself then go into the dlib
#   root folder and run:
#       python setup.py install
#   or
#       python setup.py install --yes USE_AVX_INSTRUCTIONS
#   if you have a CPU that supports AVX instructions, since this makes some
#   things run faster.  
#
#   Compiling dlib should work on any operating system so long as you have
#   CMake and boost-python installed.  On Ubuntu, this can be done easily by
#   running the command:
#       sudo apt-get install libboost-python-dev cmake
#
#   Also note that this example requires scikit-image which can be installed
#   via the command:
#       pip install scikit-image
#   Or downloaded from http://scikit-image.org/download.html. 


# Standard imports
import sys
import os
import glob
import urllib.request
import bz2

# Dependecy imports
import dlib
from skimage import io


def download_if_not_exist():
    """Download predictor trained model if not exist to detect face."""

    predictor_path = 'shape_predictor_68_face_landmarks.dat'

    if not os.path.isfile(predictor_path):

        print('Downloading from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        urllib.request.urlretrieve(
            "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
            "shape_predictor_68_face_landmarks.dat.bz2")
        print('Done')

        archive_path = 'shape_predictor_68_face_landmarks.dat.bz2'

        zipfile = bz2.BZ2File(archive_path) # open the file
        data = zipfile.read() # get the decompressed data
        open(predictor_path, 'wb').write(data) # write a uncompressed file
        del data

        # Remove archive
        os.remove(archive_path)

    return predictor_path


def detect_landmarks(dir_path):
    """Detect facial landmarks using pretrained model."""

    predictor_path = download_if_not_exist()

    detector = dlib.get_frontal_face_detector() # pylint: disable=E1101
    predictor = dlib.shape_predictor(predictor_path) # pylint: disable=E1101
    # win = dlib.image_window()

    for im_path in glob.glob(os.path.join(dir_path, "*.jpg")):

        # If file already exist don't bother to detect again
        if not os.path.isfile(im_path + '.txt'):
            print("Processing file: {}".format(im_path))
            img = io.imread(im_path)

            # win.clear_overlay()
            # win.set_image(img)

            # Ask the detector to find the bounding boxes of each face. The 1 in the
            # second argument indicates that we should upsample the image 1 time. This
            # will make everything bigger and allow us to detect more faces.
            dets = detector(img, 1)
            print("Number of faces detected: {}, choosing biggest".format(len(dets)))

            # if not detect the image
            # if len(dets) == 0:
            #     continue

            areas = []
            for det in dets:
                areas.append(det.area())

                # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                #     k, det.left(), det.top(), det.right(), det.bottom()))

            biggest_area_id = areas.index(max(areas))

            # Get the landmarks/parts for the face in box d.
            shape = predictor(img, dets[biggest_area_id])

            with open(im_path + '.txt', "a") as myfile:
                for i in range(shape.num_parts):
                    myfile.write(str(shape.part(i).x) + ' ' + str(shape.part(i).y) + '\n')

            # Draw the face landmarks on the screen.
            # win.add_overlay(shape)

            # win.add_overlay(dets)
            # dlib.hit_enter_to_continue()

if __name__ == '__main__':

    DIR_PATH = sys.argv[1]
    detect_landmarks(DIR_PATH)
