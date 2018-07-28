from PIL import ImageGrab
import os
import time


def get_current_screen(type):
	box = (0, 30, 577, 1057)
	im = ImageGrab.grab(box)
	im.save(f"{os.getcwd()}/{type}/{type}_{str(int(time.time()))}.png", "PNG")


def main():
	get_current_screen("movement")


if __name__ == "__main__": main()
