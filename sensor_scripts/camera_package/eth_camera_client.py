import socket
import time
import subprocess
import argparse

HEIGHT_DESCRIPTION = "Height of Camera (Stereo Sewn Image Applied): 480, 960, etc."
WIDTH_DESCRIPTION = "Width of Camera (Stereo Sewn Image Applied): 1280(640 e. img.), 2560 (1280), etc."
IP_DESCRIPTION = "IP of Camera Server in the Form of: 192.168.x.x"
SCRIPT_DESCRIPTION = "Input name/title of script: {script_name}.py"

def get_image_size(sock, size):
   data = sock.recv(size)
   return data.decode()

def parse_camera_msg(decoded_message):
    height = str(int(decoded_message[:4]))
    width = str(int(decoded_message[4:8]))
    print(height)
    print(width)
    cam_exposure = str(int(decoded_message[8:16]))
    print(cam_exposure)
    cam_gain = str(float(decoded_message[16:21]))
    print(cam_gain)
    cam_framerate = str(int(decoded_message[21:]))
    print(cam_framerate)
    return height, width, cam_exposure, cam_gain, cam_framerate

def wait_for_connection(host, port, retry_interval=5, max_attempts = 10):
    total_attempts = 0

    while True:
        try:
            # Create a TCP/IP socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Try to connect to the server
            sock.connect((host, port))

            data = get_image_size(sock, 23)
            params = parse_camera_msg(data)

            print(f"Successfully connected to {host}:{port} w\\ argument: size = {params[0]}:{params[1]}"
                  + f", gain = {params[2]}, exposure = {params[3]}, framerate = {params[4]}")


            return sock, params
        except (socket.error, ConnectionRefusedError) as e:
            # If the connection fails, print the error and retry after a delay
            print(f"Connection failed: {e}. Retrying in {retry_interval} seconds...")
            sock.close()
            time.sleep(retry_interval)
            total_attempts += 1
            if total_attempts > max_attempts:
                print("Could not connect to server in time. Retry Script with Server Active.")
                return -1

def run_script(script_path, params, TCP_IP):
    height, width, cam_exposure, gain, framerate = params[:]

    try:
        print(f"Running script: {script_path}")
        subprocess.run(["python", script_path, width, height, cam_exposure, gain, framerate, TCP_IP, '2'], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")

if __name__ == "__main__":

    # Instantiating Argument Parser with Description of Arguments Above
    parser = argparse.ArgumentParser(description='Setup and initialize Camera Sensor through Ethernet')
    parser.add_argument('-i', '--server', type=str, required=True, help=IP_DESCRIPTION)
    parser.add_argument('-s', '--script', type=str, required=True, help=SCRIPT_DESCRIPTION)
    args = parser.parse_args().__dict__

    # Replace with the actual server's host and port
    host = args['server']
    port = 5000

    # Replace with the path to the script you want to run after the connection
    script_to_run = args['script'] + '.py'
    
    # Wait for connection
    sock, params = wait_for_connection(host, port)

    # Boot the other script once connection is established
    run_script(script_to_run, params, args['server'])

    # Close the socket when done
    sock.close()