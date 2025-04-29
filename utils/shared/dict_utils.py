def list_of_dict_to_dict(
    list_of_dicts: list[dict], new_dict: dict = {}, depth_cnt=1
) -> dict:
    for item in list_of_dicts:
        if depth_cnt > 0:
            if isinstance(item, dict):
                new_dict.update(**item)
            else:
                depth_cnt -= 1
                new_dict = list_of_dict_to_dict(item, new_dict, depth_cnt)
        else:
            return new_dict

    return new_dict
