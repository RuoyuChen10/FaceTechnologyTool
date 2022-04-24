python main.py --dataset-root ../CelebA/Img --data-list ../CelebA/label.txt --num-classes 10177 --pre-trained ../CelebA/pretrained/ArcFace-r50-10177.pth --cls-device cuda:2 --seg-device cuda:3 --save-dir results/CelebA

python main.py --dataset-root ../VGGFace2/Img --data-list ../VGGFace2/label.txt --num-classes 8631 --pre-trained ../VGGFace2/pretrained/CosFace-r50-8631.pth --cls-device cuda:2 --seg-device cuda:3 --save-dir results/VGGFace2-CosFace


python vggface2.py --dataset-root ../VGGFace2/Img --data-list ../VGGFace2/label.txt --num-classes 8631 --backbone senet50 --pre-trained ../VGGFace2/pretrained/senet50_scratch_weight.pkl --cls-device cuda:2 --seg-device cuda:3 --save-dir results/VGGFace2-VGGFaceSeNet50