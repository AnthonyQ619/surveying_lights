import cv2
import os
import sys

args = sys.argv

num = args[1]

def get_max_num_images(home_path):
    img_list = os.listdir(home_path)

    return len(img_list)


HOME = os.path.join(os.sep, *os.getcwd().split(os.sep)[:-2], 'experiments', f'exp{num}', 'images')
#Windows Only 
if os.name == 'nt':
    HOME = HOME[:2] + '\\' + HOME[2:]
def main():
    num_images = get_max_num_images(HOME)
    for i in range(1,num_images - 1):
        if i%10 == 0:
            print(f'Image {i}/{num_images} done.')
            
        img = cv2.imread(HOME + "/frame{}.png".format(i))

        img1 = img[:, :int(img.shape[1]/2), :]
        img2 = img[:, int(img.shape[1]/2):, :]
        
        if not os.path.exists(HOME + '/image_pairs'):
            os.mkdir(HOME + '/image_pairs')
        if not os.path.exists(HOME + '/image_pairs' + '/left_camera'):
            os.mkdir(HOME + '/image_pairs' + '/left_camera')
        if not os.path.exists(HOME + '/image_pairs' + '/right_camera'):
            os.mkdir(HOME + '/image_pairs' + '/right_camera')

        cv2.imwrite(HOME + '/image_pairs' + '/left_camera'+ "/frame{}.png".format(i), img1)
        cv2.imwrite(HOME + '/image_pairs' + '/right_camera'+ "/frame{}.png".format(i), img2)


if __name__ == "__main__":
    main()
