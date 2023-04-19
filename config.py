import jax

class UNet_params:
    def __init__(self):
        self.ch = 128
        self.groups = 32
        self.scale = [1, 2, 2, 2]
        self.add_attn = [16]
        self.dropout_rate = 0.1
        self.num_res_blocks = 2
    
class config(UNet_params):
    def __init__(self):
        super().__init__()
        
        self.random_seed = 42
        
        self.dataset = 'cifar10'
        self.dataset_size = 50000
        self.data_dim = [32, 32, 3]
        self.new_dim = None
        
        self.lr = 2e-4
        self.batch = 128
        self.step = 1000000
        
        self.time_steps = 1000
        self.beta_0 = 0.0001
        self.beta_T = 0.02
        
        self.grad_clip = 1.0
        self.warmup_steps = 5000
        self.ema_decay = 0.9999
        self.rand_flip = True

        self.check_period = 10000
        # self.calc_loss_period = 500
        
        self.devices = jax.local_device_count()
        self.device_memory_threshold = 2500
        self.stats_dir = "fid_stats_cifar10_train.npz"
        self.ckpt_dir = "last_shot_2/save_no_ema"
        self.ckpt_ema_dir = "last_shot_2/save"
        self.save_dir = "last_shot_2/save"
        self.sample_dir = "last_shot_2/sample"
        self.temp_dir = "last_shot_2/temp"