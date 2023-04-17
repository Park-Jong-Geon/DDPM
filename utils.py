import os, glob, shutil
from time import ctime, time
from functools import partial
import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
import flax
from flax.training import checkpoints
from PIL import Image
from config import config

class utils(config):
    def __init__(self):
        super().__init__()
    
    @tf.autograph.experimental.do_not_convert
    def normalize(self, image, label):
        return 2 * (image / 255) - 1
    
    @tf.autograph.experimental.do_not_convert
    def augmentation(self, image, label):
        if self.rand_flip:
            image = tf.image.random_flip_left_right(image)
        return image, label
    
    def load_dataset(self, dataset, batch):
        ds = tfds.load(dataset, as_supervised=True, split='train')
        
        ds = ds.shuffle(self.dataset_size)
        
        ds = ds.repeat()
        
        ds = ds.map(self.augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(self.normalize, num_parallel_calls=tf.data.AUTOTUNE)
        if self.new_dim != None:
            ds = ds.map(lambda image: tf.image.resize(image, [*self.new_dim[0:2]]), num_parallel_calls=tf.data.AUTOTUNE)
        
        assert self.batch % self.devices == 0
        ds = ds.batch(self.batch // self.devices)
        ds = ds.batch(self.devices)
        
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        ds = map(lambda data: jax.tree_map(lambda x: x._numpy(), data), ds)
        ds = flax.jax_utils.prefetch_to_device(ds, 2)
  
        print(f"Loaded {dataset} dataset", flush=True)
        return ds

    def save_imgs(self, samples, step):
        sample_num = jnp.size(samples, axis=0)
        for i, sample in enumerate(samples):
            Image.fromarray(sample).save(f"{self.sample_dir}/{step}step_seed{self.random_seed}_{i+1}_of_{sample_num}.png")          

    def load_ckpt(self):
        assert self.ckpt_dir != None
        if not os.path.exists(self.ckpt_dir):
            raise Exception(f"Directory {dir} does not exist")
        
        if self.ckpt_ema_dir != None:
            if not os.path.exists(self.ckpt_ema_dir):
                raise Exception(f"Directory {dir} does not exist")
            return checkpoints.restore_checkpoint(ckpt_dir=self.ckpt_dir, target=self.state), checkpoints.restore_checkpoint(ckpt_dir=self.ckpt_ema_dir, target=self.state)
        else:
            return checkpoints.restore_checkpoint(ckpt_dir=self.ckpt_dir, target=self.state), None
        
    def save_ckpt(self, state, ema_state):
        assert (len(os.listdir(self.save_dir)) + 10) < 1e+8
        self.create_dir(f"{self.save_dir}_no_ema")
        
        checkpoints.save_checkpoint(ckpt_dir=f"{self.save_dir}_no_ema", target=state, step=state.step, keep=1e+8, overwrite=False)
        checkpoints.save_checkpoint(ckpt_dir=self.save_dir, target=ema_state, step=ema_state.step, keep=1e+8, overwrite=False)
        
        print(f"Checkpoint saved at {self.save_dir} and {self.save_dir}_no_ema", flush=True)
      
    def graph(self, txt_title, keyword):
        assert keyword == 'FID'
        f = open(txt_title)
        lines = f.readlines()
        f.close()
        
        values = []
        for line in lines:
            if line == '\n':
                continue
            strings = line.split()
            
            if keyword in strings:
                values.append(float(strings[-1]))

        plt.plot(values)
        plt.axis((0, len(values), min(values), max(values)))
        plt.savefig(f"{txt_title}_{keyword}_curve.png")
    
    def create_dir(self, dir):
        try:
            if not os.path.exists(dir):
                os.makedirs(dir)
        except OSError:
            print('Error: Creating directory. ' + dir)
    
    def clear_dir(self, dir):
        files = glob.glob(f"{dir}/*")
        for f in files:
            os.remove(f)
    
    def move_files(self, dir1, dir2):
        files = os.listdir(dir1)
        for f in files:
            shutil.move(os.path.join(dir1, f), dir2)
    
    def print_config(self):
        print(ctime(time()), flush=True)
        print(f"dataset {self.dataset} lr {self.lr} batch {self.batch} step {self.step}", flush=True)
        print(f"Beta scheduling : time_steps {self.time_steps} beta_0 {self.beta_0} beta_T {self.beta_T}", flush=True)
        print(f"U-Net Parameters : ch {self.ch} groups {self.groups} scale {tuple(self.scale)} add_attn {tuple(self.add_attn)} dropout_rate {self.dropout_rate} num_res_blocks {self.num_res_blocks}", flush=True)
        print(f"Learning related parameters : grad_clip {self.grad_clip} warmup_steps {self.warmup_steps} EMA decay {self.ema_decay} Random Horizontal Flip {self.rand_flip}")
        print(f"Save path : {self.save_dir}", flush=True)
        if self.ckpt_dir != None:
            print(f"Checkpoint loaded from : {self.ckpt_dir}", flush=True)
        