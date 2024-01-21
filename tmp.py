
"""
C3VD 数据集
"""

import os

scenes = [
    "cecum_t1_a", "cecum_t1_b", 
    "cecum_t2_a",
    "trans_t1_a", "trans_t1_b",
    "trans_t2_a", "trans_t2_b", "trans_t2_c",
    "trans_t3_a", "trans_t3_b",
    "trans_t4_a",
    'desc_t4_a',
    'sigmoid_t1_a', 'sigmoid_t2_a','sigmoid_t3_a','sigmoid_t3_b'
]
# weight_path, save_postfix = "log/11-26-17:22-GOLDEN/checkpoints/best.pth.tar", "" # GOLDEN
weight_path, save_postfix = "log/labs/01-04-23:08/checkpoints/4.pth.tar", "_ours_5"  # 1-norm(output)
# weight_path, save_postfix = "log/labs/01-06-00:30/checkpoints/0.pth.tar", "_inv_1" # 1/output
# weight_path, save_postfix = "log/labs/01-07-13:29/checkpoints/0.pth.tar", "_log_1" # 1/output
# scenes = [
#     'desc_t4_a',
#     'sigmoid_t1_a',
#     'sigmoid_t2_a','sigmoid_t3_a','sigmoid_t3_b'
# ]

for scene in scenes:
    print("=> processing", scene)
    cml = '''python run.py \
--model_type dpt_beit_large_512 \
--input_path /home/jiahan/jiahan/datasets/C3VD/dataset_{}_4SCDepth/scenes/scene1 \
--output_path /home/jiahan/jiahan/datasets/C3VD/MiDas_Depth_for_SC_Depth/{}_finetuning{} \
--model_weights {} \
--grayscale
'''.format(scene,scene,save_postfix,  weight_path)
    os.system(cml)


"""
USTC 真实数据集
"""
# from path import Path
# import os

# root_path = Path("/home/jiahan/jiahan/datasets/USTC/TrainValid/Images")
# save_path = Path("/home/jiahan/jiahan/datasets/USTC/TrainValid/midas_depth")
# save_path.makedirs_p()

# scenes = root_path.listdir()
# for scene in scenes:
#     if not scene.isdir(): continue
#     print("=> processing", scene)
#     save_path_tmp = save_path/str(scene).split('/')[-1]
#     save_path_tmp.makedirs_p()
#     cml = '''python run.py \
# --model_type dpt_beit_large_512 \
# --input_path {} \
# --output_path {} \
# --model_weights log/11-26-17:22-GOLDEN/checkpoints/best.pth.tar \
# --grayscale
# '''.format(scene, save_path_tmp)
#     os.system(cml)
    
    
    
    
"""
SimCol3D 数据集
"""


# from path import Path
# import os

# w_paths, s_postfix = [], []

# # weight_path, save_postfix = "log/11-26-17:22-GOLDEN/checkpoints/best.pth.tar", "_finetuning"
# # w_paths.append(weight_path)
# # s_postfix.append(save_postfix)

# weight_path, save_postfix = "log/labs/01-04-23:08/checkpoints/4.pth.tar", "_finetuning_ours_5"
# w_paths.append(weight_path)
# s_postfix.append(save_postfix)

# # weight_path, save_postfix = "log/labs/01-06-00:30/checkpoints/0.pth.tar", "_finetuning_inv_1" # 1/output
# # w_paths.append(weight_path)
# # s_postfix.append(save_postfix)

# # weight_path, save_postfix = "log/labs/01-07-13:29/checkpoints/0.pth.tar", "_finetuning_log_1" # 1/output
# # w_paths.append(weight_path)
# # s_postfix.append(save_postfix)

# for weight_path,save_postfix in zip(w_paths,s_postfix):
#     # SimCol3D
#     # III
#     root_path3 = Path("/home/jiahan/jiahan/datasets/SimCol/SyntheticColon_III/SyntheticColon_III")
#     save_path3 = Path(f"/home/jiahan/jiahan/datasets/SimCol/Midas/Midas{save_postfix}/SyntheticColon_III")
#     scenes3 = root_path3.listdir()

#     # I
#     root_path1 = Path("/home/jiahan/jiahan/datasets/SimCol/SyntheticColon_I/SyntheticColon_I")
#     save_path1 = Path(f"/home/jiahan/jiahan/datasets/SimCol/Midas/Midas{save_postfix}/SyntheticColon_I")
#     # scenes1 = [
#     #     root_path1/"Frames_S5",
#     #     root_path1/"Frames_S10",
#     #     root_path1/"Frames_S15"
#     # ]
#     scenes1 = root_path1.listdir()

#     # II
#     root_path2 = Path("/home/jiahan/jiahan/datasets/SimCol/SyntheticColon_II/SyntheticColon_II")
#     save_path2 = Path(f"/home/jiahan/jiahan/datasets/SimCol/Midas/Midas{save_postfix}/SyntheticColon_II")
#     # scenes2 = [
#     #     root_path2/"Frames_B5",
#     #     root_path2/"Frames_B10",
#     #     root_path2/"Frames_B15"
#     # ]
#     scenes2 = root_path2.listdir()

#     scenes_all = [scenes1,scenes2,scenes3]
#     save_path_all = [save_path1,save_path2,save_path3]

#     for scenes,save_path in zip(scenes_all,save_path_all):
#         # print(scenes)
#         for scene in scenes:
#             # print(scene)
#             if not scene.isdir(): continue
#             print("=> processing", scene)
#             save_path_tmp = save_path/str(scene).split('/')[-1]
#             save_path_tmp.makedirs_p()
#             cml = '''python run.py \
# --model_type dpt_beit_large_512 \
# --input_path {} \
# --output_path {} \
# --model_weights {} \
# --grayscale
# '''.format(scene, save_path_tmp, weight_path)
#             os.system(cml)
