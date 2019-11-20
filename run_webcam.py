from __future__ import print_function
from __future__ import division

import sys
sys.path.insert(0, 'src')
import argparse
import numpy as np
import transform, vgg, pdb, os
import tensorflow as tf
import cv2
from datetime import datetime
import time


models_all=[{"ckpt":"models/animalcollage_model/fns.ckpt", "style":"styles/galaxia.jpg"},
			{"ckpt":"models/plantas_model/fns.ckpt", "style":"styles/galaxia.jpg"},
			{"ckpt":"models/galaxia_model/fns.ckpt", "style":"styles/galaxia.jpg"},
			{"ckpt":"models/animal_model/fns.ckpt", "style":"styles/galaxia.jpg"},
			{"ckpt":"models/pantone_model/fns.ckpt", "style":"styles/galaxia.jpg"},
			{"ckpt":"models/stein2_model/fns.ckpt", "style":"styles/galaxia.jpg"},
			{"ckpt":"models/cristales_model/fns.ckpt", "style":"styles/galaxia.jpg"},
			{"ckpt":"models/cristales_model1/fns.ckpt", "style":"styles/galaxia.jpg"},
			{"ckpt":"models/cristales_model2/fns.ckpt", "style":"styles/galaxia.jpg"},
			{"ckpt":"models/cristales_model3/fns.ckpt", "style":"styles/galaxia.jpg"},
			{"ckpt":"models/cristales_model4/fns.ckpt", "style":"styles/galaxia.jpg"},
			{"ckpt":"models/Escher_1/fns.ckpt", "style":"styles/galaxia.jpg"},
			{"ckpt":"models/Escher_2/fns.ckpt", "style":"styles/galaxia.jpg"},
			{"ckpt":"models/Mano_1/fns.ckpt", "style":"styles/galaxia.jpg"},
			{"ckpt":"models/Mano_2/fns.ckpt", "style":"styles/galaxia.jpg"},
			{"ckpt":"models/Style_1/fns.ckpt", "style":"styles/galaxia.jpg"},
			{"ckpt":"models/Style_2/fns.ckpt", "style":"styles/galaxia.jpg"},
			{"ckpt":"models/Style_3/fns.ckpt", "style":"styles/galaxia.jpg"},
			{"ckpt":"models/Style_4/fns.ckpt", "style":"styles/galaxia.jpg"}]


models=[{"ckpt":"models/animalcollage_model/fns.ckpt", "style":"styles/galaxia.jpg"},
		{"ckpt":"models/plantas_model/fns.ckpt", "style":"styles/galaxia.jpg"},	
		{"ckpt":"models/galaxia_model/fns.ckpt", "style":"styles/galaxia.jpg"},
		{"ckpt":"models/animal_model/fns.ckpt", "style":"styles/galaxia.jpg"},
		{"ckpt":"models/pantone_model/fns.ckpt", "style":"styles/galaxia.jpg"},
		{"ckpt":"models/stein2_model/fns.ckpt", "style":"styles/galaxia.jpg"},
		{"ckpt":"models/cristales_model/fns.ckpt", "style":"styles/galaxia.jpg"},
	    {"ckpt":"models/cristales_model1/fns.ckpt", "style":"styles/galaxia.jpg"},
		{"ckpt":"models/cristales_model2/fns.ckpt", "style":"styles/galaxia.jpg"},
		{"ckpt":"models/cristales_model3/fns.ckpt", "style":"styles/galaxia.jpg"},
		{"ckpt":"models/cristales_model4/fns.ckpt", "style":"styles/galaxia.jpg"},
		{"ckpt":"models/Escher_1/fns.ckpt", "style":"styles/galaxia.jpg"},
		{"ckpt":"models/Escher_2/fns.ckpt", "style":"styles/galaxia.jpg"},
		{"ckpt":"models/Mano_1/fns.ckpt", "style":"styles/galaxia.jpg"},
		{"ckpt":"models/Mano_2/fns.ckpt", "style":"styles/galaxia.jpg"},
		{"ckpt":"models/Style_1/fns.ckpt", "style":"styles/galaxia.jpg"},
		{"ckpt":"models/Style_2/fns.ckpt", "style":"styles/galaxia.jpg"},
		{"ckpt":"models/Style_3/fns.ckpt", "style":"styles/galaxia.jpg"},
		{"ckpt":"models/Style_4/fns.ckpt", "style":"styles/galaxia.jpg"}]



# parser
parser = argparse.ArgumentParser()
parser.add_argument('--device_id', type=int, help='camera device id (default 0)', required=False, default=0)
parser.add_argument('--width', type=int, help='width to resize camera feed to (default 320)', required=False, default=640)
parser.add_argument('--disp_width', type=int, help='width to display output (default 640)', required=False, default=1200)
parser.add_argument('--disp_source', type=int, help='whether to display content and style images next to output, default 1', required=False, default=1)
parser.add_argument('--horizontal', type=int, help='whether to concatenate horizontally (1) or vertically(0)', required=False, default=1)
parser.add_argument('--num_sec', type=int, help='number of seconds to hold current model before going to next (-1 to disable)', required=False, default=-1)




def load_checkpoint(checkpoint, sess):
	saver = tf.train.Saver()
	try:
		saver.restore(sess, checkpoint)
		style = cv2.imread(checkpoint)
		return True
	except:
		print("checkpoint %s not loaded correctly" % checkpoint)
		return False


def get_camera_shape(cam):
	""" use a different syntax to get video size in OpenCV 2 and OpenCV 3 """
	cv_version_major, _, _ = cv2.__version__.split('.')
	if cv_version_major == '3' or cv_version_major == '4':
		return cam.get(cv2.CAP_PROP_FRAME_WIDTH), cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
		print("Caso 1")
	else:
		return cam.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH), cam.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
		print("Caso 2")

  
  
def make_triptych(disp_width, frame, style, output, horizontal=True):
	ch, cw, _ = frame.shape
	sh, sw, _ = style.shape
	oh, ow, _ = output.shape
	disp_height = int(disp_width * oh / ow)
	h = int(ch * disp_width * 0.5 / cw)
	w = int(cw * disp_height * 0.5 / ch)
	if horizontal:
		full_img = np.concatenate([
			cv2.resize(frame, (int(w), int(0.5*disp_height))), 
			cv2.resize(style, (int(w), int(0.5*disp_height)))], axis=0)
		full_img = np.concatenate([full_img, cv2.resize(output, (disp_width, disp_height))], axis=1)
	else:
		full_img = np.concatenate([
			cv2.resize(frame, (int(0.5 * disp_width), h)), 
			cv2.resize(style, (int(0.5 * disp_width), h))], axis=1)
		full_img = np.concatenate([full_img, cv2.resize(output, (disp_width, disp_width * oh // ow))], axis=0)
	return full_img


def main(device_id, width, disp_width, disp_source, horizontal, num_sec):
	t1 = datetime.now()
	idx_model = 0
	device_t='/gpu:0'
	g = tf.Graph()
	soft_config = tf.ConfigProto(allow_soft_placement=True)
	soft_config.gpu_options.allow_growth = True
	start_time = time.time()
	counter = 0
	x = 1
	with g.as_default(), g.device(device_t), tf.Session(config=soft_config) as sess:
		cam = cv2.VideoCapture(device_id)
		cam.set(cv2.CAP_PROP_FPS, 60)
		
		cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
		cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, 1)
		cv2.resizeWindow("frame",1920,1080)
		cam_width, cam_height = get_camera_shape(cam)
		width = width if width % 4 == 0 else width + 4 - (width % 4) # must be divisible by 4
		height = int(width * float(cam_height/cam_width))
		height = height if height % 4 == 0 else height + 4 - (height % 4) # must be divisible by 4
		img_shape = (height, width, 3)
		batch_shape = (1,) + img_shape
		print("batch shape", batch_shape)
		print("disp source is ", disp_source)
		img_placeholder = tf.placeholder(tf.float32, shape=batch_shape, name='img_placeholder')
		preds = transform.net(img_placeholder)
		
		# load checkpoint
		load_checkpoint(models[idx_model]["ckpt"], sess)
		style = cv2.imread(models[idx_model]["style"])
		
		# enter cam loop
		while True:
			ret, frame = cam.read()
			frame = cv2.resize(frame, (width, height))
			frame = cv2.flip(frame, -1)
			
			X = np.zeros(batch_shape, dtype=np.float32)
			X[0] = frame
			
			output = sess.run(preds, feed_dict={img_placeholder:X})
			output = output[:, :, :, [2,1,0]].reshape(img_shape)
			output = np.clip(output, 0, 255).astype(np.uint8)
			output = cv2.resize(output, (width, height))

			if disp_source:
				full_img = make_triptych(disp_width, frame, style, output, horizontal)
				cv2.imshow('frame', full_img)
				

			else:
				oh, ow, _ = output.shape
				output = cv2.resize(output, (1920 ,1080))
				output = cv2.rotate(output, cv2.ROTATE_90_CLOCKWISE)
				cv2.imshow('frame', output)
			

			key_ = cv2.waitKey(1)	
			if key_ == 27:
				break
			elif key_ == ord('a'):
				idx_model = (idx_model + len(models) - 1) % len(models)
				print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
				load_checkpoint(models[idx_model]["ckpt"], sess)
				style = cv2.imread(models[idx_model]["style"])
			elif key_ == ord('s'):
				idx_model = (idx_model + 1) % len(models)
				print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
				load_checkpoint(models[idx_model]["ckpt"], sess)
				style = cv2.imread(models[idx_model]["style"])

			t2 = datetime.now()
			dt = t2-t1
			if num_sec>0 and dt.seconds > num_sec:
				t1 = datetime.now()
				idx_model = (idx_model + 1) % len(models)
				print("load %d / %d : %s " % (idx_model, len(models), models[idx_model]))
				load_checkpoint(models[idx_model]["ckpt"], sess)
				style = cv2.imread(models[idx_model]["style"])

		# done
		cam.release()
		cv2.destroyAllWindows()


if __name__ == '__main__':
	opts = parser.parse_args()
	main(opts.device_id, opts.width, opts.disp_width, opts.disp_source==1, opts.horizontal==1, opts.num_sec),

