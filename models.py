import math
import numpy as np
# PyTorch
import torch
from torch import NoneType, cuda 
import torch.nn as nn
# warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
# Data science tools
from unet import UNet
from modified_unet import M_UNet 
from keras_unet import K_UNet 
import segmentation_models_pytorch as smp 
from SelfONN import SuperONN1d, SuperONN2d, SelfONN1d, SelfONN2d
from SelfONN_Pretrained_UNet import SelfONNUnet
from selfonn_unet_selflayer import SelfONNUnet_new
import SelfONN_decoders
import Unet3Plus_SMP
from SA_UNet import SA_UNet
from ACCUnet import ACC_UNet
from torchvision import models
def reset_function_generic(m):
    if hasattr(m,'reset_parameters'): 
        m.reset_parameters()

def get_pretrained_model(parentdir, model_type, model_to_load, encoder_depth, encoder_weights, q_order, max_shift, U_init_features, in_channels, out_channels, train_on_gpu, multi_gpu, decoder_attention=None, activation=None):
    dec_dict = {
        'Unet': smp.Unet,
        'UnetPlusPlus': smp.UnetPlusPlus,
        'MAnet': smp.MAnet,
        'Linknet': smp.Linknet,
        'FPN': smp.FPN,
        'PSPNet': smp.PSPNet,
        'PAN': smp.PAN,
        'DeepLabV3': smp.DeepLabV3,
        'DeepLabV3Plus': smp.DeepLabV3Plus,
        'Unet3Plus': Unet3Plus_SMP.Unet3Plus,
        'SelfONN_Unet': SelfONN_decoders.SelfONNUnet,
        'SelfONN_UnetPlusPlus': SelfONN_decoders.SelfONNUnetPlusPlus,
        'SelfONN_ResUnet': SelfONN_decoders.SelfONN_ResUnet,
        'SelfONN_FPN': SelfONN_decoders.SelfONN_FPN
    }
    
    # Define the Combined Encoder
    class CombinedEncoder(nn.Module):
        def __init__(self, encoder1, encoder2):
            super(CombinedEncoder, self).__init__()
            self.encoder1 = encoder1
            self.encoder2 = encoder2

        def forward(self, x):
            features1 = self.encoder1(x)
            features2 = self.encoder2(x)
            
            # Example: concatenate features from both encoders
            combined_features = torch.cat([features1, features2], dim=1)
            
            return combined_features

    print('Initialized with', model_type, '\n')

    if model_type == "SMP":
        enc_name, dec_name = model_to_load.split('*')

        if enc_name == 'renetDensenet':
            # Initialize individual encoders
            encoder1 = models.resnet18(pretrained=encoder_weights is not None)
            encoder2 = models.densenet121(pretrained=encoder_weights is not None).features
            
            # Remove fully connected layers
            encoder1 = nn.Sequential(*list(encoder1.children())[:-2])
            encoder2 = nn.Sequential(*list(encoder2.children())[:-1])
            
            # Create combined encoder
            combined_encoder = CombinedEncoder(encoder1, encoder2)
            
            # Determine the output channels from both encoders
            combined_feature_channels = 512 + 1024  # Example: ResNet18 (512) + DenseNet121 (1024)

            if dec_name in dec_dict:
                model = dec_dict[dec_name](
                    encoder_name='custom',
                    encoder_weights=None,
                    in_channels=in_channels,
                    classes=out_channels,
                    activation=activation,
                    decoder_attention_type=decoder_attention,
                    encoder_output_channels=combined_feature_channels
                )
                model.encoder = combined_encoder
            else:
                raise ValueError(f"Decoder '{dec_name}' is not supported.")

    elif model_type == "ONN_dec":
        enc_name, dec_name = model_to_load.split('*')
        if dec_name == 'SelfONN_Unet' or dec_name == 'SelfONN_UnetPlusPlus' or dec_name == 'SelfONN_ResUnet':
            print('decoder_attention:', str(decoder_attention), '\n')
            model = dec_dict[dec_name](enc_name, encoder_depth=encoder_depth, encoder_weights=encoder_weights, decoder_attention_type=decoder_attention, in_channels=in_channels, classes=out_channels, q_order=q_order, max_shift=max_shift, activation=activation)
        else:
            model = dec_dict[dec_name](enc_name, encoder_depth=encoder_depth, encoder_weights=encoder_weights, in_channels=in_channels, classes=out_channels, q_order=q_order, max_shift=max_shift, activation=activation)

    else:
        if model_to_load == 'UNet':  
            model = UNet(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features)
        elif model_to_load == 'M_UNet': 
            model = M_UNet(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features)
        elif model_to_load == 'K_UNet': 
            model = K_UNet(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features) 
        elif model_to_load == 'UNet_2Plus': 
            from UNet_2Plus import UNet_2Plus 
            model = UNet_2Plus(in_channels=in_channels, n_classes=out_channels)
        elif model_to_load == 'UNet_3Plus': 
            from UNet_3Plus import UNet_3Plus
            model = UNet_3Plus(in_channels=in_channels, n_classes=out_channels)
        elif model_to_load == 'MultiResUNet':
            from MultiResUNet import MultiResUNet
            model = MultiResUNet(in_features=32, alpha=1.67, in_channels=in_channels, out_channels=out_channels)
        elif model_to_load == 'CSNet':
            from csnet import CSNet
            model = CSNet(classes=out_channels, channels=in_channels)
        elif model_to_load == 'Self_UNet':
            from SelfONN_Unet import Self_UNet
            model = Self_UNet(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features, q_order=1, up_q_order=1, out_q_order=1)
        elif model_to_load == 'SelfUNet_compact':
            from SelfONN_Unet import SelfUNet_compact
            model = SelfUNet_compact(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features, q_order=1, up_q_order=1, out_q_order=1)
        elif model_to_load == 'SelfUNet_super_compact':
            from SelfONN_Unet import SelfUNet_super_compact
            model = SelfUNet_super_compact(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features, q_order=1, up_q_order=1, out_q_order=1)
        elif model_to_load == 'SuperUNet_super_compact':
            from SelfONN_Unet import SuperUNet_super_compact
            model = SuperUNet_super_compact(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features, q_order=1, up_q_order=1, out_q_order=1)
        elif model_to_load == 'Super_FPN':
            from SelfONN_Unet import Super_FPN
            model = Super_FPN(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features, q_order=3, up_q_order=1, out_q_order=1)
        elif model_to_load == 'Attention_Super_FPN':
            from SelfONN_Unet import Attention_Super_FPN
            model = Attention_Super_FPN(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features, q_order=1, up_q_order=1, out_q_order=1)
        elif model_to_load == 'Densenet121 SuperONN_Unet':
            model = SelfONNUnet('densenet121', decoder_attention_type=decoder_attention, in_channels=3, classes=1, q_order=q_order, max_shift=max_shift, activation='logsoftmax')
        elif model_to_load == 'SelfONN_Unet_new':
            model = SelfONNUnet_new('densenet121', decoder_attention_type=decoder_attention, in_channels=3, classes=1, q_order=3, max_shift=0, activation=activation)
        elif model_to_load == 'SA_UNet':
            model = SA_UNet(in_channels=in_channels, num_classes=out_channels)
        elif model_to_load == 'ACC_UNet':
            model = ACC_UNet(n_channels=in_channels, n_classes=out_channels, n_filts=U_init_features)

    # Reset parameters (optional)
    # reset_fn = reset_function_generic
    # model.apply(reset_fn)

    # Move to GPU and parallelize
    if train_on_gpu:
        model = model.to('cuda')
    if multi_gpu:
        model = nn.DataParallel(model)
        
    return model


# def get_pretrained_model(parentdir, model_type, model_to_load, encoder_depth, encoder_weights, q_order, max_shift, U_init_features,in_channels,out_channels,train_on_gpu,multi_gpu, decoder_attention = None, activation = None):
#     dec_dict = {
#         'Unet': smp.Unet,
#         'UnetPlusPlus' : smp.UnetPlusPlus,
#         'MAnet' : smp.MAnet,
#         'Linknet' : smp.Linknet,
#         'FPN' : smp.FPN,
#         'PSPNet' : smp.PSPNet,
#         'PAN' : smp.PAN,
#         'DeepLabV3' : smp.DeepLabV3,
#         'DeepLabV3Plus' : smp.DeepLabV3Plus,
#         'Unet3Plus' : Unet3Plus_SMP.Unet3Plus,
#         'SelfONN_Unet' : SelfONN_decoders.SelfONNUnet,
#         'SelfONN_UnetPlusPlus' : SelfONN_decoders.SelfONNUnetPlusPlus,
#         'SelfONN_ResUnet' : SelfONN_decoders.SelfONN_ResUnet,
#         'SelfONN_FPN' : SelfONN_decoders.SelfONN_FPN
#     }


#     #loading models from the SMP library
#     print('Initialized with', model_type,'\n')
#     if model_type == "SMP":
#         enc_name, dec_name = model_to_load.split('*')
#         if dec_name == 'Unet' or dec_name == 'UnetPlusPlus' or dec_name == 'Unet3Plus':
#           print('decoder_attention:', str(decoder_attention), '\n')
#           model = dec_dict[dec_name](enc_name, encoder_depth = encoder_depth, encoder_weights = encoder_weights, decoder_attention_type = decoder_attention, in_channels=in_channels, classes=out_channels, activation=activation)
#         else:
#           if dec_name == 'PAN':
#             model = dec_dict[dec_name](enc_name, encoder_weights = encoder_weights, in_channels=in_channels, classes=out_channels, activation=activation)
#           else:
#             model = dec_dict[dec_name](enc_name, encoder_depth = encoder_depth, encoder_weights = encoder_weights, in_channels=in_channels, classes=out_channels, activation=activation)

#     #loading models from the ONN Decoders file
#     elif model_type == "ONN_dec":
#         enc_name, dec_name = model_to_load.split('*')
#         if dec_name == 'SelfONN_Unet' or dec_name == 'SelfONN_UnetPlusPlus' or dec_name == 'SelfONN_ResUnet':
#             print('decoder_attention:', str(decoder_attention), '\n')
#             model = dec_dict[dec_name](enc_name, encoder_depth = encoder_depth, encoder_weights = encoder_weights, decoder_attention_type = decoder_attention, in_channels=in_channels, classes=out_channels,  q_order = q_order, max_shift = max_shift, activation=activation)
#         else:
#             model = dec_dict[dec_name](enc_name, encoder_depth = encoder_depth, encoder_weights = encoder_weights, in_channels=in_channels, classes=out_channels,  q_order = q_order, max_shift = max_shift, activation=activation)
            
#     #loading custom built models        
#     else:
#         if model_to_load == 'UNet':  
#             model = UNet(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features)
#         elif model_to_load == 'M_UNet': 
#             model = M_UNet(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features)
#         elif model_to_load == 'K_UNet': 
#             model = K_UNet(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features) 
#         elif model_to_load == 'UNet_2Plus': 
#             from UNet_2Plus import UNet_2Plus 
#             model = UNet_2Plus(in_channels=in_channels, n_classes=out_channels)
#         elif model_to_load == 'UNet_3Plus': 
#             from UNet_3Plus import UNet_3Plus
#             model = UNet_3Plus(in_channels=in_channels, n_classes=out_channels)
#         elif model_to_load == 'MultiResUNet':
#             from MultiResUNet import MultiResUNet
#             model = MultiResUNet(in_features=32, alpha=1.67, in_channels=in_channels, out_channels=out_channels)

#         elif model_to_load == 'CSNet':
#             from csnet import CSNet
#             model = CSNet(classes=out_channels , channels=in_channels)


#         elif model_to_load == 'Self_UNet':
#             from SelfONN_Unet import Self_UNet
#             model = Self_UNet(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features, q_order=1, up_q_order=1, out_q_order=1)
#         elif model_to_load == 'SelfUNet_compact':
#             from SelfONN_Unet import SelfUNet_compact
#             model = SelfUNet_compact(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features, q_order=1, up_q_order=1, out_q_order=1)
#         elif model_to_load == 'SelfUNet_super_compact':
#             from SelfONN_Unet import SelfUNet_super_compact
#             model = SelfUNet_super_compact(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features, q_order=1, up_q_order=1, out_q_order=1)
#         elif model_to_load == 'SuperUNet_super_compact':
#             from SelfONN_Unet import SuperUNet_super_compact
#             model = SuperUNet_super_compact(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features, q_order=1, up_q_order=1, out_q_order=1)
#         elif model_to_load == 'Super_FPN':
#             from SelfONN_Unet import Super_FPN
#             model = Super_FPN(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features, q_order=3, up_q_order=1, out_q_order=1)
#         elif model_to_load == 'Attention_Super_FPN':
#             from SelfONN_Unet import Attention_Super_FPN
#             model = Attention_Super_FPN(in_channels=in_channels, out_channels=out_channels, init_features=U_init_features, q_order=1, up_q_order=1, out_q_order=1)
#         elif model_to_load == 'Densenet121 SuperONN_Unet':
#             model = SelfONNUnet('densenet121', decoder_attention_type = decoder_attention, in_channels=3, classes=1, q_order = q_order, max_shift = max_shift, activation='logsoftmax')
#         elif model_to_load == 'SelfONN_Unet_new':
#             model = SelfONNUnet_new('densenet121', decoder_attention_type = decoder_attention, in_channels=3, classes=1, q_order = 3, max_shift = 0, activation=activation)
#         elif model_to_load == 'SA_UNet':
#             model = SA_UNet(in_channels=in_channels, num_classes=out_channels)
#         elif model_to_load == 'ACC_UNet':
#             model = ACC_UNet(n_channels=in_channels, n_classes=out_channels, n_filts=U_init_features)



#     # reset_fn = reset_function_generic 
#     # model.apply(reset_fn) 

#     # Move to gpu and parallelize
#     if train_on_gpu:
#         model = model.to('cuda')
#     if multi_gpu:
#         model = nn.DataParallel(model)
#     return model 