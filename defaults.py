from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()

_C.device = "cuda"
_C.rff_flag = True
_C.nolabel = False



_C.kernel_x = "RFFGaussian"

_C.kernel_x_options = CN()
_C.kernel_x_options.rff_dim = 1000
_C.kernel_x_options.sigma_numel_max = 4000


_C.kernel_s = "RFFGaussian"

_C.kernel_s_options = CN()


_C.kernel_z = "RFFGaussian"

_C.kernel_z_options = CN()

_C.kernel_z = "RFFGaussian"

_C.kernel_z_options = CN()
_C.kernel_z_options.rff_dim = 1000
_C.kernel_z_options.sigma_numel_max = 4000

_C.kernel_y = "RFFGaussian"

_C.kernel_y_options = CN()

# ---------------------------------------------------------#
#           metric_control
# ---------------------------------------------------------#

_C.metric_control = CN()
_C.metric_control.HSIC = "NonParametricDependence"
_C.metric_control.KCC = "NonParametricDependence"

# ---------------------------------------------------------#
#           metric_control_options
# ---------------------------------------------------------#
_C.metric_control_options = CN()
_C.metric_control_options.HSIC = CN()
_C.metric_control_options.KCC = CN()

_C.metric_control_options.HSIC.rff = 1
_C.metric_control_options.HSIC.score_list = CN()
_C.metric_control_options.HSIC.score_list.HSIC = CN()
_C.metric_control_options.HSIC.kernel_z = "RFFGaussian"
_C.metric_control_options.HSIC.kernel_z_opts = CN()
_C.metric_control_options.HSIC.kernel_z_opts.rff_dim = 1000
_C.metric_control_options.HSIC.kernel_z_opts.sigma_numel_max = 3000
_C.metric_control_options.HSIC.kernel_s = "RFFGaussian"
_C.metric_control_options.HSIC.kernel_s_opts = CN()
_C.metric_control_options.HSIC.kernel_s_opts.rff_dim = 1000
_C.metric_control_options.HSIC.kernel_s_opts.sigma_numel_max = 3000

_C.metric_control_options.KCC.rff = 1
_C.metric_control_options.KCC.score_list = CN()
_C.metric_control_options.KCC.score_list.KCC = CN()
_C.metric_control_options.KCC.score_list.KCC.lam = 0.001
_C.metric_control_options.KCC.kernel_z = "RFFGaussian"
_C.metric_control_options.KCC.kernel_z_opts = CN()
_C.metric_control_options.KCC.kernel_z_opts.rff_dim = 1000
_C.metric_control_options.KCC.kernel_z_opts.sigma_numel_max = 3000
_C.metric_control_options.KCC.kernel_s = "RFFGaussian"
_C.metric_control_options.KCC.kernel_s_opts = CN()
_C.metric_control_options.KCC.kernel_s_opts.rff_dim = 1000
_C.metric_control_options.KCC.kernel_s_opts.sigma_numel_max = 3000


_C.dim_z = 768


_C.seed = 2



_C.tau_i = 0.0

_C.tau_z_i = 0.0

_C.tau_t = 0.0

_C.tau_z_t = 0.0

_C.sample_ratio1 = 1.0

_C.sample_ratio2 = 1.0

_C.gamma_i =3e-6#5e-5#3e-6# celeba 5e-5
_C.gamma_t =3e-6#5e-5#3e-6# celeba 5e-5

_C.load_base_model = "clip_ViTL14"
_C.dataset = "waterbirds"
_C.debias = True

_C.bs_trn = 128
_C.bs_val = 128

_C.iters = 1