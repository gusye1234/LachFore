from segmentation_driven_pose.utils import *
from segmentation_driven_pose.segpose_net import SegPoseNet
import world
from time import time
import os

class Seg:
    def __init__(self, weight, intrinsics):
        if weight not in ["linemod", "ycb"]:
            raise TypeError('unsupported dataset \'%s\'.' % weight)
        self.weight = weight
        self.intrinsics = np.array([[intrinsics.fx,.00,intrinsics.ppx],
                                    [0.0,intrinsics.fy, intrinsics.ppy],
                                    [0.0,0.0,1.0]])
        if weight == "linemod":
            data = "./segmentation_driven_pose/data/data-LINEMOD.cfg"
            weightfile = "./segmentation_driven_pose/model/occluded-linemod.pth"
            self.names = ['ape', 'can', 'cat', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher']
            self.vertes = np.load('./segmentation_driven_pose/data/Occluded-LINEMOD/LINEMOD_vertex.npy')
            
        else:
            data = "./segmentation_driven_pose/data/data-YCB.cfg" 
            weightfile = "./segmentation_driven_pose/model/ycb-video.pth"
            self.names = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',
                        '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',
                        '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',
                        '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']
            self.vertes = np.load('./segmentation_driven_pose/data/YCB-Video/YCB_vertex.npy')
        data_option = read_data_cfg(data)
        self.model = SegPoseNet(data_option)
        if world.model_verbose:
            self.model.print_network()
        self.model.load_weights(weightfile)
        if world.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.model.cuda()
        self.model.eval()
        print(">LOADING WEIGHT DONE (%s)" % (weight))
    def predict(self, image, draw=False):
        start = time()
        print(">PREDICT BEGIN")

        pred = do_detect(self.model,
                        image, self.intrinsics,
                        world.bestCnt, world.conf_thresh, world.use_gpu)
        print("predict %d objects" % (len(pred)))
        pred_out = Seg.pred_filter(pred)
        print(" ".join([self.names[obj[0]] for obj in pred]))
        print(">END(duration: %.2f)" % (time()-start))
        if draw:
            return pred_out, visualize_predictions(pred, image, self.vertes, self.intrinsics)
        return pred_out, None

    @staticmethod
    def pred_filter(prediction):
        ans = []
        for i in prediction:
            ans.append(i[:2])
        return ans




