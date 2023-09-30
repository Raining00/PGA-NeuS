# this file is written for dynamic rendering, should lock down the parameters in the network
# forward returns the render result, and backward returns the gard
# need to copy data from trained model in init function
# better to use R-T transform when rendering a single ray, make easier to adapt in the future

class Dynamic_rigid_renderer:
    def __init__(self, rgb_net, sdf_net):
        self.rgb_net = rgb_net
        self.sdf_net = sdf_net

        return
    

    # TODO: render after rotate rays_o & d
    def forward(self, rays_o, rays_d, near, far, pose_mat, R, T, perturb_overwrite=-1, background_rgb=None, cos_anneal_ratio=0.0):
        # make new rays_o & d
         
        # generate sample points as original neus renderer

        # returns the rendered rgb, img loss & grads of R & T ?        
        return
    
    def render_at(self):
        # make new rays_o & d
         
        # generate sample points as original

        # returns the rendered rgb, img loss & grads of R & T ?
        return


    def backward(self):
        return
