# We only support manually setting the bounding box of first frame and save the results in debug directory.
# We only support manually setting the bounding box of first frame and save the results in debug directory.

##########-------------- MixFormerV2-Base-----------------##########
# python tracking/video_demo.py \
#   mixformer2_vit_online \
#   288_depth8_score \
#   xxx.mp4  \
#   --optional_box [YOUR_X] [YOUR_Y] [YOUR_W] [YOUR_H] \ 
#   --params__model models/mixformerv2_base.pth.tar --debug 1 \
#   --params__search_area_scale 4.5 --params__update_interval 25 --params__online_size 1


#   /mnt/sda1/videos/compare/infrared/20230215/MOV_20230215_171959960_tracking_0.mp4  \
#   /mnt/sda1/videos/case_seq_bak/105.mp4 \
#   ~/tmp/videos/case_seq_bak/105.mp4 \
#   "/mnt/NAS/飞行部-第二空间/安防-当月上传飞行数据/20230815 延庆照明吊舱/video/MOV_20230815_175543633.mp4"\
#   --show_results \

# python3 tracking/video_demo.py mixformer2_vit_online 288_depth8_score \
#   /mnt/sda1/videos/case_seq_bak/105.mp4 \
# --show_results \
#   --save_results \
#   --params__model ./models/mixformerv2_base.pth.tar --debug 0 \
#   --params__search_area_scale 3 --params__update_interval 10000 --params__online_sizes 1 \
# #   --optional_box 1188	138	116	114

# /mnt/sda1/videos/20230815/cut/20230815/MOV_20230215_171332185_tracking_3.mp4 \

python3 tracking/video_demo.py mixformer2_vit_online 224_depth4_mlp1_score \
  /mnt/sda1/videos/case_seq_bak/011.mp4 \
  --show_results \
  --save_results \
  --params__model ./models/mixformerv2_small.pth.tar --debug 0 \
  --params__search_area_scale 2 --params__update_interval 10000 --params__online_sizes 1 \
#   --optional_box 1188	138	116	114


# python3 tracking/video_demo.py mixformer2_vit_online 224_depth4_mlp1_score \
#   ~/tmp/videos/case_seq_bak/105.mp4 \
#   --save_results \
#   --params__model ./models/mixformerv2_small.pth.tar --debug 0 \
#   --params__search_area_scale 3 --params__update_interval 10000 --params__online_sizes 1 \
#   --optional_box 1188	138	116	114