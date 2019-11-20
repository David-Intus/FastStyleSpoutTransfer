from __future__ import print_function
from __future__ import division
import sys
sys.path.insert(0, 'src')
import argparse
import tensorflow as tf
import numpy as np
import transform, vgg, pdb, os
import time
import cv2
import SpoutSDK
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *
from OpenGL.GLU import *

def parse_args():
    desc = "Webcam and Spout implementation to Stylr transfer"

    parse = argparse.ArgumentParser(description = desc)
    
    parse.add_argument("--style_model", type= str, default='models/Escher_1/fns.ckpt', help='Direccion de los chekpoints (.ckpt)')
    
    parse.add_argument("--spout_size", nargs= 2, type= int, default= [640,480], help= "Largo y ancho del spout a mandar")
    
    parse.add_argument("--windows_size", nargs= 2, type= int, default=[1024,720], help= "Tamaño de la ventana")

    parse.add_argument("--camID", type=int, default=0, help="Webcam Index")
    
    return parse.parse_args()

def main():
    args = parse_args()

    width = args.windows_size[0]
    height = args.windows_size[1]
    display = (width, height)

    pygame.init()
    pygame.display.set_caption('Style transfer and Spout for Python')
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    pygame.display.gl_set_attribute(pygame.GL_ALPHA_SIZE,8)

    cap = cv2.VideoCapture(0)
    cap.set(3,width)
    cap.set(4,height)

    glMatrixMode(GL_PROJECTION)
    glOrtho(0,width,height,0,1,-1)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)
    glClearColor(0.0,0.0,0.0,0.0)
    glEnable(GL_TEXTURE_2D)
   
    spoutSender = SpoutSDK.SpoutSender()
    spoutSenderWidth = width
    spoutSenderHeight = height
    spoutSender.CreateSender('Spout for Python Webcam Sender Example', width, height, 0)
    
    senderTextureID = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D, senderTextureID)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glBindTexture(GL_TEXTURE_2D, 0)

    device_t='/gpu:0'
    g = tf.Graph()
    soft_config = tf.ConfigProto(allow_soft_placement=True)
    soft_config.gpu_options.allow_growth = True

    with g.as_default(), g.device(device_t), tf.Session(config=soft_config) as sess:
        img_shape = (height, width, 3)
        batch_shape = (1,) + img_shape
        style = tf.placeholder(tf.float32, shape = batch_shape, name = 'input')

        preds = transform.net(style)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        try:
            saver.restore(sess, args.style_model)
            #style = cv2.imread(args.style_model)
        except:
            print("checkpoint %s not loaded correctly" % args.style_model)

        while(True):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    spoutReceiver.ReleaseReceiver()
                    pygame.quit()
                    quit()
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            glBindTexture(GL_TEXTURE_2D, senderTextureID)
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, frame )
            data = glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, outputType = None)
            glBindTexture(GL_TEXTURE_2D, 0)        
            data.shape = (data.shape[1], data.shape[0], data.shape[2])

            X = np.zeros(batch_shape, dtype=np.float32)
            X[0] = frame 
            output = sess.run(preds, feed_dict={style:X})
            output = output[:,:,:, [2,1,0]].reshape(img_shape)
            output = np.clip(output, 0.0, 255.0)
            output = output.astype(np.uint8)
            glBindTexture(GL_TEXTURE_2D, senderTextureID)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            # copy output into texture
            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, spoutSenderWidth, spoutSenderHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, output)

            spoutSender.SendTexture(senderTextureID, GL_TEXTURE_2D, spoutSenderWidth, spoutSenderHeight, False, 0)
            # Clear screen
            glClear(GL_COLOR_BUFFER_BIT  | GL_DEPTH_BUFFER_BIT )
            """# reset the drawing perspective
            glLoadIdentity()
        
            # Draw texture to screen
            glBegin(GL_QUADS)

            glTexCoord(0,0)        
            glVertex2f(0,0)

            glTexCoord(1,0)
            glVertex2f(width,0)

            glTexCoord(1,1)
            glVertex2f(width,height)

            glTexCoord(0,1)
            glVertex2f(0,height)

            glEnd()

            # update window
            pygame.display.flip()          """   
        
            # unbind our sender texture
            glBindTexture(GL_TEXTURE_2D, 0)

if __name__ == '__main__':
    main()
 


        