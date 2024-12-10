from Brf_light_analysis.brf_estimation import *
from light_bulb_detection.bulb_detection import LightSourceDetector, BulbSourceAnalyzer

# Helper Function
def video_length(file_name):
    cap = cv2.VideoCapture(file_name)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return length

f_n = "C:\\Users\\Anthony\\Documents\\Projects\\LCG_Experiments\\BRF_video_data\\Experiment2\\f"

def read_video_cv2(file_name, n_frames=80):
    cap = cv2.VideoCapture(file_name)
    all = []
    i = 0
    while cap.isOpened() and i < n_frames:
        ret, frame = cap.read()
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        arr = np.array(gray_frame)
        all.append(arr)
        i += 1
    return np.array(all)

def main():
    for i in range(1, 192):
        file_name = f_n + str(i) + ".avi"
        print(i, ")", video_length(file_name))

    
    file_name = f_n + "34.avi"

    vid = read_video_cv2(file_name, 55)
    vid2 = read_video_cv2(file_name, 110)
    vid2_data = vid2[55:]
    print(vid.shape)

    ld = BulbSourceAnalyzer(maxArea=150000, minArea=200, brf_filter=False, debug = False)
    kp1, brf_points1, brf_roi1 = ld.detect_lightSources(vid, n_frame=1, roi_offset=10)
    kp2, brf_points2, brf_roi2 = ld.detect_lightSources(vid2_data, n_frame=1, roi_offset=10)


    extraction_methods = ["CoIntensity","Pruned pixels", "Circular region"]
    px_radius = 10
    x_offset = 5
    y_offset = 0
    for i in range(len(kp1)):
        topL_pt1, botR_pt1 = brf_roi1[i][0], brf_roi1[i][1]
        estimated_anlys_pt1 = brf_points1[i]
        topL_pt2, botR_pt2 = brf_roi2[i][0], brf_roi2[i][1]
        estimated_anlys_pt2 = brf_points2[i]

        brf_signals = compute_raw_brf_signal(vid, topL_pt1, botR_pt1, px_radius, x_offset, y_offset, extraction_methods, estimated_anlys_pt1)
        x_offset = 4
        brf_signals = compute_raw_brf_signal(vid2_data, topL_pt2, botR_pt2, px_radius, x_offset, y_offset, extraction_methods, estimated_anlys_pt2)
        

main()