from DDPM import DDPM

if __name__ == "__main__":
        _DDPM = DDPM()
    
        _DDPM.print_config()
    
        _DDPM.create_dir(_DDPM.save_dir)
        _DDPM.create_dir(_DDPM.sample_dir)
        _DDPM.create_dir(_DDPM.temp_dir)
        
        ckpt = None
        ckpt_ema = None
        if _DDPM.ckpt_dir != None:
                ckpt, ckpt_ema = _DDPM.load_ckpt()
        '''
        ds = _DDPM.load_dataset(_DDPM.dataset, _DDPM.batch)
        
        _DDPM.train(ds, ckpt, ckpt_ema)
        if _DDPM.ckpt_dir == None:
                _DDPM.ckpt_ema_dir = _DDPM.save_dir
                _DDPM.ckpt_dir = f"{_DDPM.save_dir}_no_ema"
        trained_ckpt, trained_ckpt_ema = _DDPM.load_ckpt()
        
        _DDPM.sample(trained_ckpt_ema, (_DDPM.dataset_size//_DDPM.devices)*_DDPM.devices)
        '''
        _DDPM.sample(ckpt_ema, 8)
        # print(_DDPM.fid.calculate_fid(_DDPM.stats_dir, _DDPM.sample_dir))