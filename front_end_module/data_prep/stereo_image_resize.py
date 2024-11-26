import cv2
import os
import sys

args = sys.argv

height = args[0]
width = args[1]
num = args[2]

def get_max_num_images(home_path):
    img_list = os.listdir(home_path)
    count_list = list()
    for img in img_list:
        img_name = img.split('.')[0]
        img_count = int(img_name.split('_')[-1])
        count_list.append(img_count)

    return max(count_list)


HOME = os.path.join(os.sep, *os.getcwd().split(os.sep)[:-1], 'experiments', f'exp{num}', 'images')
#Windows Only
if os.name == 'nt':
    HOME = HOME[:2] + '\\' + HOME[2:]
def main():
    num_images = get_max_num_images(HOME)
    for i in range(1,num_images+1):
        if i%10 == 0:
            print(f'Image {i}/{num_images} done.')
            
        img = cv2.imread(HOME + "/frame{}.png".format(i))

        img1 = img[:, :int(img.shape[1]/2), :]
        img2 = img[:, int(img.shape[1]/2):, :]

        img1 = cv2.flip(img1, 1)
        img2 = cv2.flip(img2, 1)
        
        if not os.path.exists(HOME + '/image_pairs'):
            os.mkdir(HOME + '/image_pairs')
        if not os.path.exists(HOME + '/image_pairs' + '/left_camera'):
            os.mkdir(HOME + '/image_pairs' + '/left_camera')
        if not os.path.exists(HOME + '/image_pairs' + '/right_camera'):
            os.mkdir(HOME + '/image_pairs' + '/right_camera')

        cv2.imwrite(HOME + '/image_pairs' + '/right_camera'+ "/pair_Left{}.png".format(i), img1)
        cv2.imwrite(HOME + '/image_pairs' + '/right_camera'+ "/pair_Right{}.png".format(i), img2)


if __name__ == "__main__":
    main()
