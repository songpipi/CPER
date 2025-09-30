# -*- coding: utf-8 -*-
import argparse


def parse_opt():
    parser = argparse.ArgumentParser()

    #### training params
    parser.add_argument('--fixed_encoder', type=bool, default=True, help='if True, fix the params of image encoder')
    parser.add_argument('--load_from_epoch',type=str, default=None, help="epoch number")
    parser.add_argument("--eval_steps",type=int, default=2000, help="evaluate model per k training steps")
    parser.add_argument("--stop_after_evals", type=int, default=10, help="If there is no increase in performance for K evaluations, the training is terminated.")
    parser.add_argument("--len_prefix", type=int, default=1,help="the length of prefix")
    parser.add_argument("--prompt_text",type=str,default="the emotion is")
    parser.add_argument("--max_seq_len",type=int,default=30,help="the length of full explanation")
    parser.add_argument("--prefix_dim", type=int, default=768)
    parser.add_argument("--num_train_epochs", type=int, default=25)
    parser.add_argument("--alpha", type=float, default=1.0, help="the weight of XE loss")
    parser.add_argument("--learning_rate",type=float,default=2e-5, help="the LR of decoder")
    parser.add_argument("--en_LR",type=float,default=2e-5)
    parser.add_argument("--emo_en_LR",type=float,default=4e-5)
    parser.add_argument("--batch_size",type=int,default=32)

    ### Emotion Encoder Configuration
    parser.add_argument("--emotion_encoder_type", type=str, default="linear",
                       choices=["linear", "mlp", "transformer"],
                       help="Type of emotion encoder: linear (original), mlp (current default), transformer (new)")
    parser.add_argument("--transformer_layers", type=int, default=2,
                       help="Number of transformer layers for transformer emotion encoder")
    parser.add_argument("--transformer_heads", type=int, default=8,
                       help="Number of attention heads for transformer emotion encoder")


    parser.add_argument("--ckpt_path",type=str,default='art1-bucket7/ckpts/')
    parser.add_argument("--caption_save_path",type=str,default="art1-bucket7/")
    parser.add_argument("--nle_data_train_path",type=str,default='artemis-master/type2/artEmisX_cl_train.json')
    parser.add_argument("--annFileExp",type=str,default='artemis-master/type2/artEmisX_test_annot_exp.json')
    parser.add_argument("--nle_data_val_path",type=str,default='artemis-master/type2/artEmisX_test.json')

    ### bucket
    parser.add_argument("--bucket_path",type=str,default='artemis-master/type2/bucket_uniform.pickle')
    parser.add_argument("--control_signal_at_inference",type=int,default=2)

    args = parser.parse_args()
    
    return args
