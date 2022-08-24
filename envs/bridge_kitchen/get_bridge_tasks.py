from envs.bridge_kitchen.dataset_config_real import *
from envs.bridge_kitchen.toykitchen_pickplace_dataset import *

def get_bridge_tasks(dataset, target_dataset):
    if dataset == 'single_task':
        train_tasks = train_dataset_single_task
        eval_tasks = eval_dataset_single_task
    elif dataset =='11tasks':
        print("using 11 tasks")
        train_tasks = train_dataset_11_task
        eval_tasks = eval_dataset_11_task
    elif dataset == 'tk1_pickplace':
        train_tasks, eval_tasks = get_toykitchen1_pickplace()
    elif dataset == 'tk2_pickplace':
        train_tasks, eval_tasks = get_toykitchen2_pickplace()
    elif dataset == 'all_pickplace':
        train_tasks, eval_tasks = get_all_pickplace()
    elif dataset == 'open_micro_single':
        train_tasks = train_dataset_single_task_openmicro
        eval_tasks = eval_dataset_single_task_openmicro
    elif dataset == 'online_reaching_pixels':
        train_tasks = online_reaching_pixels
        eval_tasks = online_reaching_pixels_val
    elif dataset == 'online_reaching_pixels_first100':
        train_tasks = online_reaching_pixels_first100
        eval_tasks = online_reaching_pixels_val_first100
    elif dataset == 'toykitchen1_pickplace':
        train_tasks, eval_tasks = get_toykitchen1_pickplace()
    elif dataset == 'toykitchen2_pickplace':
        train_tasks, eval_tasks = get_toykitchen2_pickplace()
    elif dataset == 'all_pickplace':
        train_tasks, eval_tasks = get_all_pickplace()
    elif dataset == 'all_pickplace_except_tk6':
        train_tasks, eval_tasks = get_all_pickplace_exclude_tk6()
    elif dataset == 'toykitchen2_pickplace_simpler':
        train_tasks, eval_tasks = get_toykitchen2_pickplace_cardboardfence_reversible_simple()
    elif dataset == 'toykitchen6_knife_in_pot':
        train_tasks, eval_tasks = get_toykitchen6_knife_in_pot()
    elif dataset == 'toykitchen6_croissant_out_of_pot':
        train_tasks, eval_tasks = get_toykitchen6_croissant_out_of_pot()
    elif dataset == 'toykitchen6_pear_from_plate':
        train_tasks, eval_tasks = get_toyktichen6_pear_from_plate()
    elif dataset == 'toykitchen6_sweet_potato_on_plate':
        train_tasks, eval_tasks = get_toykitchen6_put_sweet_potato_on_plate()
    elif dataset == 'toykitchen6_sweet_potato_in_bowl':
        train_tasks, eval_tasks = get_toykitchen6_put_sweet_potato_in_bowl()
    elif dataset == 'toykitchen6_lime_in_pan_sink':
        train_tasks, eval_tasks = get_toyktichen6_put_lime_in_pan_sink()
    elif dataset == 'toykitchen6_drumstick_on_plate':
        train_tasks, eval_tasks = get_toykitchen6_put_drumstick_on_plate()
    elif dataset == 'toykitchen6_cucumber_in_pot':
        train_tasks, eval_tasks = get_toykitchen6_cucumber_in_orange_pot()
    elif dataset == 'toykitchen6_carrot_in_pan':
        train_tasks, eval_tasks = get_toykitchen6_carrot_in_pan()
    elif dataset == 'debug':
        num_debug_tasks = 3
        train_tasks, eval_tasks = get_all_pickplace_exclude_tk6()
        train_tasks = train_tasks[:num_debug_tasks]
        eval_tasks = eval_tasks[:num_debug_tasks]
    else:
        raise ValueError('dataset not found! ' + dataset)

    if target_dataset != '':
        if target_dataset == 'toykitchen2_pickplace_cardboardfence_reversible':
            target_train_tasks, target_eval_tasks = get_toykitchen2_pickplace_cardboardfence_reversible()
            # target_train_tasks, target_eval_tasks = get_toykitchen2_pickplace_cardboardfence_reversible_simple()
        elif target_dataset == 'toykitchen2_pickplace_simpler':
            target_train_tasks, target_eval_tasks = get_toykitchen2_pickplace_cardboardfence_reversible_simple()
        elif target_dataset == 'toykitchen6_pickplace_reversible':
            target_train_tasks, target_eval_tasks = get_toykitchen6_pickplace_reversible()
        elif target_dataset == 'toykitchen6_target_domain':
            target_train_tasks, target_eval_tasks = get_toykitchen6_target_domain()
        elif target_dataset == 'toykitchen6_new_target_domain':
            target_train_tasks, target_eval_tasks = get_toykitchen6_new_target_domain()
        elif target_dataset == 'toykitchen6_target_domain_two_tasks':
            target_train_tasks, target_eval_tasks = get_toykitchen6_new_target_domain_2_tasks()
        elif target_dataset == 'toykitchen6_target_domain_five_tasks':
            target_train_tasks, target_eval_tasks = get_toykitchen6_new_target_domain_5_tasks()
        elif target_dataset == 'toykitchen6_knife_in_pot':
            target_train_tasks, target_eval_tasks = get_toykitchen6_knife_in_pot()
        elif target_dataset == 'toykitchen6_croissant_out_of_pot':
            target_train_tasks, target_eval_tasks = get_toykitchen6_croissant_out_of_pot()
        elif target_dataset == 'toykitchen6_pear_from_plate':
            target_train_tasks, target_eval_tasks = get_toyktichen6_pear_from_plate()
        elif target_dataset == 'toykitchen6_sweet_potato_on_plate':
            target_train_tasks, target_eval_tasks = get_toykitchen6_put_sweet_potato_on_plate()
        elif target_dataset == 'toykitchen6_sweet_potato_in_bowl':
            target_train_tasks, target_eval_tasks = get_toykitchen6_put_sweet_potato_in_bowl()
        elif target_dataset == 'toykitchen6_lime_in_pan_sink':
            target_train_tasks, target_eval_tasks = get_toyktichen6_put_lime_in_pan_sink()
        elif target_dataset == 'toykitchen6_drumstick_on_plate':
            target_train_tasks, target_eval_tasks = get_toykitchen6_put_drumstick_on_plate()
        elif target_dataset == 'toykitchen6_cucumber_in_pot':
            target_train_tasks, target_eval_tasks = get_toykitchen6_cucumber_in_orange_pot()
        elif target_dataset == 'toykitchen6_carrot_in_pan':
            target_train_tasks, target_eval_tasks = get_toykitchen6_carrot_in_pan()
        elif target_dataset == 'toykitchen6_big_corn_in_big_pot':
            target_train_tasks, target_eval_tasks = get_toykitchen6_big_corn_in_big_pot()
        elif target_dataset == 'toykitchen1_pickplace_cardboardfence_reversible':
            target_train_tasks, target_eval_tasks = get_toykitchen1_pickplace_cardboardfence_reversible()
        elif target_dataset == 'toykitchen2_sushi_targetdomain':
            target_train_tasks, target_eval_tasks = get_toykitchen2_sushi_targetdomain()
        elif target_dataset == 'debug':
            num_debug_tasks=1
            target_train_tasks, target_eval_tasks = get_toykitchen2_pickplace_cardboardfence_reversible_simple()
            target_train_tasks = target_train_tasks[:num_debug_tasks]
            target_eval_tasks = target_eval_tasks[:num_debug_tasks]
        else:
            raise ValueError('target dataset not found! ' + target_dataset)
    else:
        target_train_tasks = []
        target_eval_tasks = []

    return train_tasks, eval_tasks, target_train_tasks, target_eval_tasks