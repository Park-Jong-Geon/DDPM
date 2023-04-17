from jax_fid.fid import FID

_FID = FID(156800)

print(_FID.calculate_fid('fid_stats_cifar10_train.npz', 'last_shot/sample'))