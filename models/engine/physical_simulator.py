from models.engine.rigid_body_torch import RigidBodySimulator

class PhysicalSimulator:
    def __init__(self, options):
        self.rigid_simulator = RigidBodySimulator(options=options)
        self.delta_frame = options['delta_frame']
        self.frame = options["frames"]
        self.substeps = options["substep"]
    

    def forward(self, f, query_func):
        if f > 0:
            # for j in range(self.delta_frame):
            for i in range(self.substeps * (f - 1), self.substeps * f):
                self.rigid_simulator(i, query_func)
            translation, quaternion = self.rigid_simulator.get_transformInfo(i)
            self.rigid_simulator.export_mesh(i)
            return translation, quaternion
        
    def set_init_quaternion(self, init_quaternion):
        self.rigid_simulator.set_init_quaternion(init_quaternion=init_quaternion)
    
    def set_init_translation(self, init_translation):
        self.rigid_simulator.set_init_translation(init_translation=init_translation)
    
    def get_mu(self):
        return self.rigid_simulator.get_mu()
    
    