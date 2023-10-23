import torch


def merge_template_search(inp_list, return_search=False, return_template=False):
    """NOTICE: search region related features must be in the last place"""
    seq_dict = {"feat": torch.cat([x["feat"] for x in inp_list], dim=0),
                "mask": torch.cat([x["mask"] for x in inp_list], dim=1),
                "pos": torch.cat([x["pos"] for x in inp_list], dim=0)}
    if return_search:
        x = inp_list[-1]
        seq_dict.update({"feat_x": x["feat"], "mask_x": x["mask"], "pos_x": x["pos"]})
    if return_template:
        z = inp_list[0]
        seq_dict.update({"feat_z": z["feat"], "mask_z": z["mask"], "pos_z": z["pos"]})
    return seq_dict


def get_qkv(inp_list):

    dic_temp = inp_list[0]
    src_temp_8 = dic_temp["feat_8"]
    pos_temp_8 = dic_temp["pos_8"]
    src_temp_16 = dic_temp["feat_16"]
    pos_temp_16 = dic_temp["pos_16"]
    dic_dytemp = inp_list[1]
    dy_temp_8 =  dic_dytemp["feat_8"]
    dy_temp_16 = dic_dytemp["feat_16"]


    dic_search = inp_list[-1]
    src_search_8 =  dic_search["feat_8"]
    pos_search_8 = dic_search["pos_8"]
    src_search_16 = dic_search["feat_16"]
    pos_search_16 = dic_search["pos_16"]


    return src_temp_8, pos_temp_8, src_temp_16, pos_temp_16, src_search_8, pos_search_8, src_search_16 ,pos_search_16,dy_temp_8,dy_temp_16
