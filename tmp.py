import json 

if __name__ == '__main__':
    input_path = "./debug/236.json"
    out_path = "./debug/out2.json"
    rt_list, r_list, j_list = None, None, None
    with open(input_path, 'r') as json_file:
        rt_list = json.load(json_file)
    r_list, t_list = rt_list['out_r'], rt_list['out_t']
    write_out_dict = {}
    image_count = 21
    for i in range(0, image_count):
        write_out_dict[str(i) + '_R'] = r_list[i]
        write_out_dict[str(i) + '_T'] = t_list[i]
        
    with open(out_path, 'w') as json_file:
        json.dump(write_out_dict, json_file, indent=4)  
        
        