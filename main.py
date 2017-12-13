import os, pprint, time
import numpy as np
import tensorflow as tf
import logging 
import logging.handlers
from glob import glob
from random import shuffle
from model import generate, encode, decode
from utils import save_images, save_image, save_npz, get_image, denorm_save_image
from config import get_config
pp = pprint.PrettyPrinter()

base_dir =None

def readx(f_in, n_line):
    l_x = list()
    n_loop=0
    
    with open(f_in,'r') as file:    
        for line in file:
            n_loop +=1
            if(n_loop > n_line):
                break
            else:
                l_x.append(np.fromstring(line, dtype=float, sep=','))
    file.close()        
    
    return np.asarray(l_x)  

def readz(f_in, n_line):
    l_z = list()
    n_loop=0
    
    with open(f_in,'r') as file:    
        for line in file:
            n_loop +=1
            if(n_loop > n_line):
                break
            else:
                l_z.append(np.fromstring(line, dtype=np.float32, sep=','))
    file.close()        
    return np.asarray(l_z)  
                  
def writez(conf, n_img):
    z_fix = np.random.uniform(-1, 1, size=(n_img, conf.n_z))
    with open(os.path.join(base_dir, 'z.csv'),'w') as file:
        for i in range(len(z_fix)):
            file.write(str(z_fix[i].tolist()).replace("[", "").replace("]", "")+ '\n')
    file.close()
    
def writex(conf, n_img):
    data_files = glob(os.path.join(conf.data_dir, conf.dataset, "*"))
    shuffle(data_files)
    l_f = data_files[0:n_img]
    l_x=[get_image(f, conf.n_img_pix, is_crop=conf.is_crop, resize_w=conf.n_img_out_pix, is_grayscale = conf.is_gray) for f in l_f]
    
    with open(os.path.join(base_dir, 'x.csv'),'w') as file:
        for i in range(len(l_x)):
            file.write(str(l_x[i].tolist()).replace("[", "").replace("]", "")+ '\n')
    file.close()
            
    x_fix = np.array(l_x).astype(np.float32)

    if(conf.is_gray == 1):
        x_fix = x_fix.reshape(conf.n_batch,conf.n_img_out_pix, conf.n_img_out_pix,1 )

    x_fix= np.clip((x_fix + 1)*127.5, 0, 255)
    save_image(x_fix,'{}/x_fix.png'.format(base_dir))

def loadWeight(sess, conf):
        
    saver = tf.train.Saver()
    saver.restore(sess, os.path.join(conf.load_dir, conf.ckpt_nm))#tf.train.latest_checkpoint(conf.load_dir))

def main(conf):


    logger = logging.getLogger("desc") 
    logger.setLevel(logging.INFO) 
    fileHandler = logging.FileHandler(os.path.join(base_dir, 'log.txt')) 
    logger.addHandler(fileHandler) 
    #streamHandler = logging.StreamHandler()
    #logger.addHandler(streamHandler) 

    if conf.is_gray :
        n_channel=1
    else:
        n_channel=3

    # init directories
    checkpoint_dir = os.path.join(base_dir,conf.curr_time)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
          
    ##========================= DEFINE MODEL ===========================##
    #z = tf.random_uniform(conf.n_batch, conf.n_z), minval=-1.0, maxval=1.0)
    z = readz(os.path.join(base_dir, 'z.csv'), conf.n_batch)
    
    x_net =  tf.placeholder(tf.float32, [conf.n_batch, conf.n_img_pix, conf.n_img_pix, n_channel], name='real_images')

    k_t = tf.Variable(0., trainable=False, name='k_t')

    # execute generator
    g_net, g_vars, g_conv = generate(z, conf.n_img_out_pix, conf.n_conv_hidden, n_channel,  is_train=True, reuse=False)
        
    # execute discriminator
    e_g_net, enc_vars, e_g_conv = encode(g_net, conf.n_z, conf.n_img_out_pix, conf.n_conv_hidden,  is_train=True, reuse=False)
    d_g_net, dec_vars, d_g_conv = decode(e_g_net, conf.n_z, conf.n_img_out_pix, conf.n_conv_hidden, n_channel,  is_train=True, reuse=False)
    
    e_x_net, _, e_x_conv = encode(x_net, conf.n_z, conf.n_img_out_pix, conf.n_conv_hidden,  is_train=True, reuse=True)
    d_x_net, _, d_x_conv = decode(e_x_net, conf.n_z, conf.n_img_out_pix, conf.n_conv_hidden, n_channel, is_train=True, reuse=True)
    
    g_img=tf.clip_by_value((g_net + 1)*127.5, 0, 255)
    #x_img=tf.clip_by_value((x_net + 1)*127.5, 0, 255)
    d_g_img=tf.clip_by_value((d_g_net + 1)*127.5, 0, 255)
    d_x_img=tf.clip_by_value((d_x_net + 1)*127.5, 0, 255)
    
    d_vars = enc_vars + dec_vars

    d_loss_g = tf.reduce_mean(tf.abs(d_g_net - g_net))
    d_loss_x = tf.reduce_mean(tf.abs(d_x_net - x_net))
    
    d_loss= d_loss_x - k_t * d_loss_g
    g_loss = tf.reduce_mean(tf.abs(d_g_net - g_net))
    
    d_loss_prev = d_loss
    g_loss_prev = g_loss
    k_t_prev = k_t

    g_optim = tf.train.AdamOptimizer(conf.g_lr).minimize(g_loss, var_list=g_vars)
    d_optim = tf.train.AdamOptimizer(conf.d_lr).minimize(d_loss, var_list=d_vars)

    balance = conf.gamma * d_loss_x - g_loss
    measure = d_loss_x + tf.abs(balance)

    with tf.control_dependencies([d_optim, g_optim]):
        k_update = tf.assign(k_t, tf.clip_by_value(k_t + conf.lambda_k * balance, 0, 1))
   
    # start session
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)
    
    loadWeight(sess, conf)
    
    x_fix = readx(os.path.join(base_dir, 'x.csv'), conf.n_batch)
    x_fix = x_fix.reshape(conf.n_batch,conf.n_conv_hidden, conf.n_img_out_pix,n_channel ) 
    
    
    n_loop=2
    for itr in range(n_loop):

        fetch_dict = {
            "kupdate": k_update,
            "m": measure,
            "b" : balance,
            'gnet':g_net,
            'dgnet': d_g_net,
            'dxnet':d_x_net,
            'xnet':x_net,
            'gconv':g_conv,
            'egconv':e_g_conv,
            'dgconv':d_g_conv,
            'exconv':e_x_conv,
            'dxconv':d_x_conv,
            "dlossx": d_loss_x,
            "gloss": g_loss,
            "dloss": d_loss,
            "kt": k_t,
            'gimg':g_img,
            'dgimg': d_g_img,
            'dximg':d_x_img,
        }
        
        result = sess.run(fetch_dict, feed_dict={x_net:x_fix})
        
        logger.info('measure: '+str(result['m']))
        logger.info('balance: '+str(result['b']))
        logger.info('gloss: '+str(result['gloss']))
        logger.info('dloss: '+str(result['dloss']))
        logger.info('dlossx: '+str(result['dlossx']))
        logger.info('k_t: '+str(result['kt']))
        
        
        if itr==0:
            
            gconv = result['gconv']
            for i in range(len(gconv)):
                conv= np.clip((gconv[i] + 1)*127.5, 0, 255)
                s,h,w,c = conv.shape
                for j in range(c):
                   c_img = conv[:,:,:,j:j+1]
                   save_image(c_img, os.path.join(checkpoint_dir, 'gen_'+str(i)+'_'+str(j)+'_'+str(h)+'_conv.png'))
                   
            dgconv = result['dgconv']
            for i in range(len(dgconv)):
                conv= np.clip((dgconv[i] + 1)*127.5, 0, 255)
                s,h,w,c = conv.shape
                for j in range(c):
                   c_img = conv[:,:,:,j:j+1]
                   save_image(c_img, os.path.join(checkpoint_dir, 'dec_g_'+str(i)+'_'+str(j)+'_'+str(h)+'_conv.png'))
            
            dxconv = result['dxconv']
            for i in range(len(dxconv)):
                conv= np.clip((dxconv[i] + 1)*127.5, 0, 255)
                s,h,w,c = conv.shape
                for j in range(c):
                   c_img = conv[:,:,:,j:j+1]
                   save_image(c_img, os.path.join(checkpoint_dir, 'dec_x_'+str(i)+'_'+str(j)+'_'+str(h)+'_conv.png'))     
                          
            exconv = result['exconv']
            for i in range(len(exconv)):
                conv= np.clip((exconv[i] + 1)*127.5, 0, 255)
                s,h,w,c = conv.shape
                for j in range(c):
                   c_img = conv[:,:,:,j:j+1]
                   save_image(c_img, os.path.join(checkpoint_dir, 'enc_x_'+str(i)+'_'+str(j)+'_'+str(h)+'_conv.png'))

            egconv = result['egconv']
            for i in range(len(egconv)):
                conv= np.clip((egconv[i] + 1)*127.5, 0, 255)
                s,h,w,c = conv.shape
                for j in range(c):
                   c_img = conv[:,:,:,j:j+1]
                   save_image(c_img, os.path.join(checkpoint_dir, 'enc_g_'+ str(i)+'_'+str(j)+'_'+str(h)+'_conv.png'))
                                      
   
            
        gnet = result['gnet']
        dgnet = result['dgnet']
        dxnet = result['dxnet']
        xnet = result['xnet']
        for i in range(conf.n_batch):
            logger.info('g_net: '+ str(gnet[i].tolist()).replace("[", "").replace("]", ""))
            logger.info('d_g_net: '+ str(dgnet[i].tolist()).replace("[", "").replace("]", ""))
            logger.info('x_net: '+ str(xnet[i].tolist()).replace("[", "").replace("]", ""))
            logger.info('d_x_net: '+ str(dxnet[i].tolist()).replace("[", "").replace("]", ""))
        
    
                
        gimg = result['gimg']
        dgimg = result['dgimg']
        dximg = result['dximg']
        save_image(gimg, os.path.join(checkpoint_dir, str(itr)+'_final_g_img.png'))
        save_image(dgimg, os.path.join(checkpoint_dir, str(itr)+'_final_d_g_img.png'))
        save_image(dximg, os.path.join(checkpoint_dir, str(itr)+'_final_d_x_img.png'))
        
    sess.close()

if __name__ == '__main__':
    #load configuration
    conf, _ = get_config()
    pp.pprint(conf)
    
    base_dir = os.path.join(conf.log_dir, 'desc')
    #writex(conf, 64)
    #writez(conf, 64)
    main(conf)
