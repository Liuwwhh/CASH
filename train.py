import time
import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
from load_data import get_data, get_data_Cross_Label
from utils import calc_map_k, f_quantifyloss, f_Coarse_grained_similarity_loss, bit_balance_loss_pm1

loss_MSE = torch.nn.MSELoss()

def train_model(args, log, 
                hashing_model, 
                input_data_par, dataloader, 
                task_index, 
                result_dict, checkpoints_path, 
                f_Fine_grained_similarity_loss, 
                history_database_code_list_image, 
                history_database_code_list_text,
                ):
    """
    Train the model for a specific task index.
    """
    if task_index != 0:
        hashing_model.load(checkpoints_path)
    
    hashing_model.task_classifier.add_new_task()
    hashing_model.add_prompt()
    hashing_model.eval()
    hashing_model.cuda()

    if task_index == 0:
        if args.prompt_mode == 'separate':
            optimizer = optim.AdamW(
                [
                    {'params': hashing_model.image_net_fine.parameters(), 'lr': args.learning_rate},
                    {'params': hashing_model.image_net_coarse.parameters(), 'lr': args.learning_rate},
                    {'params': hashing_model.text_net_fine.parameters(), 'lr': args.learning_rate},
                    {'params': hashing_model.text_net_coarse.parameters(), 'lr': args.learning_rate},
                    {'params': hashing_model.mask_net.parameters(), 'lr': args.learning_rate},
                    {'params': hashing_model.image_prompt_list[-1], 'lr': args.learning_rate},
                    {'params': hashing_model.text_prompt_list[-1], 'lr': args.learning_rate},
                    {'params': hashing_model.attention.parameters(), 'lr': args.learning_rate},
                    {'params': hashing_model.task_classifier.get_active_classifier().parameters(), 'lr': args.learning_rate},
                ]
            )
        elif args.prompt_mode == 'share':
            optimizer = optim.AdamW(
                [
                    {'params': hashing_model.image_net_fine.parameters(), 'lr': args.learning_rate},
                    {'params': hashing_model.image_net_coarse.parameters(), 'lr': args.learning_rate},
                    {'params': hashing_model.text_net_fine.parameters(), 'lr': args.learning_rate},
                    {'params': hashing_model.text_net_coarse.parameters(), 'lr': args.learning_rate},
                    {'params': hashing_model.mask_net.parameters(), 'lr': args.learning_rate},
                    {'params': hashing_model.prompt_list[-1], 'lr': args.learning_rate},
                    {'params': hashing_model.attention.parameters(), 'lr': args.learning_rate},
                    {'params': hashing_model.task_classifier.get_active_classifier().parameters(), 'lr': args.learning_rate},
                ]
            )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
    else:
        if args.prompt_mode == 'separate':
            optimizer = optim.AdamW(
                [
                    {'params': hashing_model.image_net_fine.parameters(), 'lr': args.extend_learning_rate},
                    {'params': hashing_model.image_net_coarse.parameters(), 'lr': args.extend_learning_rate},
                    {'params': hashing_model.text_net_fine.parameters(), 'lr': args.extend_learning_rate},
                    {'params': hashing_model.text_net_coarse.parameters(), 'lr': args.extend_learning_rate},
                    {'params': hashing_model.mask_net.parameters(), 'lr': args.extend_learning_rate},
                    {'params': hashing_model.image_prompt_list[-1], 'lr': args.extend_learning_rate},
                    {'params': hashing_model.text_prompt_list[-1], 'lr': args.extend_learning_rate},
                    {'params': hashing_model.attention.parameters(), 'lr': args.extend_learning_rate},
                    {'params': hashing_model.task_classifier.get_active_classifier().parameters(), 'lr': args.extend_learning_rate},
                ]
            )
        elif args.prompt_mode == 'share':
            optimizer = optim.AdamW(
                [
                    {'params': hashing_model.image_net_fine.parameters(), 'lr': args.extend_learning_rate},
                    {'params': hashing_model.image_net_coarse.parameters(), 'lr': args.extend_learning_rate},
                    {'params': hashing_model.text_net_fine.parameters(), 'lr': args.extend_learning_rate},
                    {'params': hashing_model.text_net_coarse.parameters(), 'lr': args.extend_learning_rate},
                    {'params': hashing_model.mask_net.parameters(), 'lr': args.extend_learning_rate},
                    {'params': hashing_model.prompt_list[-1], 'lr': args.extend_learning_rate},
                    {'params': hashing_model.attention.parameters(), 'lr': args.extend_learning_rate},
                    {'params': hashing_model.task_classifier.get_active_classifier().parameters(), 'lr': args.extend_learning_rate},
                ]
            )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

    max_mapi2t = max_mapt2i = torch.zeros(1, dtype=torch.float32)
    max_mapi2t = max_mapi2t.cuda()
    max_mapt2i = max_mapt2i.cuda()
    max_rBX = None
    max_rBY = None
    train_dataloader = dataloader['train']
    number_train = input_data_par['number_train']
    all_label = input_data_par['train_label'].type(torch.float)

    if task_index == 0:
        image_buffer = torch.randn(number_train, args.bit)
        text_buffer = torch.randn(number_train, args.bit)
        image_buffer_coarse = torch.randn(number_train, args.bit)
        image_buffer_fine = torch.randn(number_train, args.bit)
        text_buffer_coarse = torch.randn(number_train, args.bit)
        text_buffer_fine = torch.randn(number_train, args.bit)
        image_buffer = image_buffer.cuda()
        text_buffer = text_buffer.cuda()
        image_buffer_coarse = image_buffer_coarse.cuda()
        image_buffer_fine = image_buffer_fine.cuda()
        text_buffer_coarse = text_buffer_coarse.cuda()
        text_buffer_fine = text_buffer_fine.cuda()
        for epoch in tqdm(range(args.epoch)):
            hashing_model.train()
            for i, batch in enumerate(train_dataloader):
                # Random Index
                index = np.random.permutation(number_train)
                ind = index[0: args.batch_size]

                # batch data
                label = input_data_par['train_label'][ind, :].type(torch.float).cuda()
                image = input_data_par['train_image'][ind, :].type(torch.float).cuda()
                text = input_data_par['train_text'][ind, :].type(torch.float).cuda()

                image_code_fine, image_code_coarse, task_distinctiveness_image, mask_code, text_code_fine, text_code_coarse, task_distinctiveness_text, image_feature, text_feature = hashing_model.train_prompt_enhance(image, text)
                
                # Distillation encoding
                distill_image_code_fine = hashing_model.text_net_fine(image_feature)
                distill_image_code_coarse = hashing_model.text_net_coarse(image_feature)
                distill_text_code_fine = hashing_model.image_net_fine(text_feature)
                distill_text_code_coarse = hashing_model.image_net_coarse(text_feature)

                # Coarse-fine granularity combination
                image_hash = torch.tanh(image_code_fine) * ((1-mask_code)/2) + torch.tanh(image_code_coarse) * ((1+mask_code)/2)
                text_hash = torch.tanh(text_code_fine) * ((1-mask_code)/2) + torch.tanh(text_code_coarse) * ((1+mask_code)/2)
                distill_image_hash = torch.tanh(distill_image_code_fine) * ((1-mask_code)/2) + torch.tanh(distill_image_code_coarse) * ((1+mask_code)/2)
                distill_text_hash = torch.tanh(distill_text_code_fine) * ((1-mask_code)/2) + torch.tanh(distill_text_code_coarse) * ((1+mask_code)/2)

                # Update hash codes
                image_buffer[ind, :] = image_hash.data
                text_buffer[ind, :] = text_hash.data
                image_buffer_fine[ind, :] = image_code_fine.data
                text_buffer_fine[ind, :] = text_code_fine.data
                image_buffer_coarse[ind, :] = image_code_coarse.data
                text_buffer_coarse[ind, :] = text_code_coarse.data

                # Loss calculation

                # 1. Quantization loss + bit balance loss
                quantify_loss = f_quantifyloss(image_hash) + f_quantifyloss(text_hash) + f_quantifyloss(mask_code) \
                    + bit_balance_loss_pm1(image_hash) + bit_balance_loss_pm1(text_hash) + bit_balance_loss_pm1(mask_code)
                quantify_loss = quantify_loss * args.quantify_loss

                # 2. Fine-grained hash code loss
                fine_loss = f_Fine_grained_similarity_loss(image_code_fine, text_code_fine) \
                    + f_Fine_grained_similarity_loss(image_hash, text_hash)
                fine_loss = fine_loss * args.fine_loss

                # 3. Coarse-grained similarity loss
                coarse_loss = f_Coarse_grained_similarity_loss(label, all_label, image_code_coarse, image_buffer_coarse) \
                            + f_Coarse_grained_similarity_loss(label, all_label, text_code_coarse, text_buffer_coarse) \
                            + f_Coarse_grained_similarity_loss(label, all_label, image_hash, image_buffer) \
                            + f_Coarse_grained_similarity_loss(label, all_label, text_hash, text_buffer)
                coarse_loss = coarse_loss * args.coarse_loss

                # 4. Task distinctiveness loss
                task_distinct_loss = (torch.ones(task_distinctiveness_image.shape).cuda() - task_distinctiveness_image).mean() + (torch.ones(task_distinctiveness_text.shape).cuda() - task_distinctiveness_text).mean()
                task_distinct_loss = task_distinct_loss * args.task_distinct_loss

                # 5. Distillation loss
                distill_loss = loss_MSE(image_hash, distill_image_hash) + loss_MSE(text_hash, distill_text_hash) \
                                + loss_MSE(image_code_fine, distill_image_code_fine) + loss_MSE(text_code_fine, distill_text_code_fine) \
                                + loss_MSE(image_code_coarse, distill_image_code_coarse) + loss_MSE(text_code_coarse, distill_text_code_coarse)
                distill_loss = distill_loss * args.distill_loss

                # Total loss
                loss = quantify_loss + fine_loss + coarse_loss + task_distinct_loss + distill_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            lr_scheduler.step()

            # Validation
            if epoch % args.valid_epoch == 0:
                if args.valid:
                    hashing_model.eval()
                    with torch.no_grad():
                        separation = "single"
                        mapi2t_1000, mapt2i_1000, rBX, rBY = valid_fun(hashing_model, args, 
                                                    input_data_par['test_image'], input_data_par['database_image'],
                                                    input_data_par['test_text'], input_data_par['database_text'],
                                                    input_data_par['test_label'], input_data_par['database_label'],
                                                    separation, task_index, top_k=1000)
                    log.info('...epoch: %3d, valid MAP: MAP_1000(i->t): %3.4f, MAP_1000(t->i): %3.4f' % (epoch + 1, mapi2t_1000, mapt2i_1000))
                    if mapi2t_1000+mapt2i_1000 > max_mapi2t+max_mapt2i:
                        max_mapi2t = mapi2t_1000
                        max_mapt2i = mapt2i_1000
                        hashing_model.save(checkpoints_path)
                        torch.save(image_buffer.cpu(), args.image_old_dataset_hash_code_path)
                        torch.save(text_buffer.cpu(), args.text_old_dataset_hash_code_path)
                        max_rBX = rBX
                        max_rBY = rBY

    else:
        old_image_hash = torch.load(args.image_old_dataset_hash_code_path).cuda()
        old_text_hash = torch.load(args.text_old_dataset_hash_code_path).cuda()
        image_buffer = torch.randn(number_train, args.bit)
        text_buffer = torch.randn(number_train, args.bit)
        image_buffer_coarse = torch.randn(number_train, args.bit)
        image_buffer_fine = torch.randn(number_train, args.bit)
        text_buffer_coarse = torch.randn(number_train, args.bit)
        text_buffer_fine = torch.randn(number_train, args.bit)
        image_buffer = image_buffer.cuda()
        text_buffer = text_buffer.cuda()
        image_buffer_coarse = image_buffer_coarse.cuda()
        image_buffer_fine = image_buffer_fine.cuda()
        text_buffer_coarse = text_buffer_coarse.cuda()
        text_buffer_fine = text_buffer_fine.cuda()
        _, _, old_label, _, _, _, _, _, _ = get_data(args.dataset_path, task_index-1, args.dataset_name)
        old_label = old_label.type(torch.float).cuda()
        old_similarity = (all_label @ old_label.t() > 0).type(torch.float).cuda()
        for epoch in tqdm(range(args.epoch)):
            hashing_model.train()
            for i, batch in enumerate(train_dataloader):
                # Random Index
                index = np.random.permutation(number_train)
                ind = index[0: args.batch_size]

                # batch data
                label = input_data_par['train_label'][ind, :].type(torch.float).cuda()
                image = input_data_par['train_image'][ind, :].type(torch.float).cuda()
                text = input_data_par['train_text'][ind, :].type(torch.float).cuda()

                # get hash code
                image_code_fine, image_code_coarse, task_distinctiveness_image, mask_code, text_code_fine, text_code_coarse, task_distinctiveness_text, image_feature, text_feature = hashing_model.train_prompt_enhance(image, text)

                # Distillation encoding
                distill_image_code_fine = hashing_model.text_net_fine(image_feature)
                distill_image_code_coarse = hashing_model.text_net_coarse(image_feature)
                distill_text_code_fine = hashing_model.image_net_fine(text_feature)
                distill_text_code_coarse = hashing_model.image_net_coarse(text_feature)

                # Coarse-fine granularity combination
                image_hash = torch.tanh(image_code_fine) * ((1-mask_code)/2) + torch.tanh(image_code_coarse) * ((1+mask_code)/2)
                text_hash = torch.tanh(text_code_fine) * ((1-mask_code)/2) + torch.tanh(text_code_coarse) * ((1+mask_code)/2)
                distill_image_hash = torch.tanh(distill_image_code_fine) * ((1-mask_code)/2) + torch.tanh(distill_image_code_coarse) * ((1+mask_code)/2)
                distill_text_hash = torch.tanh(distill_text_code_fine) * ((1-mask_code)/2) + torch.tanh(distill_text_code_coarse) * ((1+mask_code)/2)

                # Update hash codes
                image_buffer[ind, :] = image_hash.data
                text_buffer[ind, :] = text_hash.data
                image_buffer_fine[ind, :] = image_code_fine.data
                text_buffer_fine[ind, :] = text_code_fine.data
                image_buffer_coarse[ind, :] = image_code_coarse.data
                text_buffer_coarse[ind, :] = text_code_coarse.data

                # Loss calculation

                # 1. Quantization loss + bit balance loss
                quantify_loss = f_quantifyloss(image_hash) + f_quantifyloss(text_hash) + f_quantifyloss(mask_code) \
                    + bit_balance_loss_pm1(image_hash) + bit_balance_loss_pm1(text_hash) + bit_balance_loss_pm1(mask_code)
                quantify_loss = quantify_loss * args.quantify_loss

                # 2. Fine-grained hash code loss
                fine_loss = f_Fine_grained_similarity_loss(image_code_fine, text_code_fine) \
                    + f_Fine_grained_similarity_loss(image_hash, text_hash)
                fine_loss = fine_loss * args.fine_loss

                # 3. Coarse-grained similarity loss
                coarse_loss = f_Coarse_grained_similarity_loss(label, all_label, image_code_coarse, image_buffer_coarse) \
                            + f_Coarse_grained_similarity_loss(label, all_label, text_code_coarse, text_buffer_coarse) \
                            + f_Coarse_grained_similarity_loss(label, all_label, image_hash, image_buffer) \
                            + f_Coarse_grained_similarity_loss(label, all_label, text_hash, text_buffer)
                coarse_loss = coarse_loss * args.coarse_loss

                # 4. Exclude loss
                exclude_loss = F.mse_loss(image_hash @ old_image_hash.t(),
                                        args.bit * old_similarity[ind, :],
                                        reduction='mean') \
                            + F.mse_loss(text_hash  @ old_text_hash.t(),
                                        args.bit * old_similarity[ind, :],
                                        reduction='mean')
                exclude_loss = exclude_loss * args.exclude_loss

                # 5. Task distinctiveness loss
                task_distinct_loss = (torch.ones(task_distinctiveness_image.shape).cuda() - task_distinctiveness_image).mean() + (torch.ones(task_distinctiveness_text.shape).cuda() - task_distinctiveness_text).mean()
                task_distinct_loss = task_distinct_loss * args.task_distinct_loss

                # 6. Distillation loss
                distill_loss = loss_MSE(image_hash, distill_image_hash) + loss_MSE(text_hash, distill_text_hash) \
                            + loss_MSE(image_code_fine, distill_image_code_fine) + loss_MSE(text_code_fine, distill_text_code_fine) \
                            + loss_MSE(image_code_coarse, distill_image_code_coarse) + loss_MSE(text_code_coarse, distill_text_code_coarse)
                distill_loss = distill_loss * args.distill_loss

                # Total loss
                loss = quantify_loss + fine_loss + coarse_loss + task_distinct_loss + exclude_loss + distill_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            lr_scheduler.step()

            # Validation
            if epoch % args.valid_epoch == 0:
                if args.valid:
                    hashing_model.eval()
                    with torch.no_grad():
                        separation = "single"
                        mapi2t_1000, mapt2i_1000, rBX, rBY = valid_fun(hashing_model, args, 
                                                    input_data_par['test_image'], input_data_par['database_image'],
                                                    input_data_par['test_text'], input_data_par['database_text'],
                                                    input_data_par['test_label'], input_data_par['database_label'],
                                                    separation, task_index, top_k=1000)
                    log.info('...epoch: %3d, valid MAP: MAP_1000(i->t): %3.4f, MAP_1000(t->i): %3.4f' % (epoch + 1, mapi2t_1000, mapt2i_1000))
                    if mapi2t_1000+mapt2i_1000 > max_mapi2t+max_mapt2i:
                        max_mapi2t = mapi2t_1000
                        max_mapt2i = mapt2i_1000
                        hashing_model.save(checkpoints_path)
                        torch.save(image_buffer.cpu(), args.image_old_dataset_hash_code_path)
                        torch.save(text_buffer.cpu(), args.text_old_dataset_hash_code_path)
                        max_rBX = rBX
                        max_rBY = rBY

    history_database_code_list_image.append(max_rBX)
    history_database_code_list_text.append(max_rBY)

    result_dict['I2T'][task_index, task_index] = max_mapi2t.cpu().numpy()
    result_dict['T2I'][task_index, task_index] = max_mapt2i.cpu().numpy()
    log.info('...test MAP: MAP_1000(i->t): %3.4f, MAP_1000(t->i): %3.4f' % (max_mapi2t, max_mapt2i))

    print(f'Start evaluation!')
    hashing_model.load(checkpoints_path)
    hashing_model.cuda()
    hashing_model.eval()

    if task_index == 0:
        pass
    else:
        if not args.old_dataset_code_is_useful:
            start_time_all = time.time()

            test_combined_image = []
            test_combined_text = []
            test_combined_label = []

            database_combined_image = []
            database_combined_text = []
            database_combined_label = []
            
            for i in range(task_index+1):
                _, _, _, test_image, test_text, test_label, database_image, database_text, database_label = get_data(args.dataset_path, i, args.dataset_name)
                test_combined_image.append(test_image)
                test_combined_text.append(test_text)
                test_combined_label.append(test_label)
                database_combined_image.append(database_image)
                database_combined_text.append(database_text)
                database_combined_label.append(database_label)

            test_image = torch.from_numpy(np.concatenate(test_combined_image)).cuda()
            test_text = torch.from_numpy(np.concatenate(test_combined_text)).cuda()
            test_label = torch.from_numpy(np.concatenate(test_combined_label)).cuda()

            database_image = torch.from_numpy(np.concatenate(database_combined_image)).cuda()
            database_text = torch.from_numpy(np.concatenate(database_combined_text)).cuda()
            database_label = torch.from_numpy(np.concatenate(database_combined_label)).cuda()
            
            with torch.no_grad():
                separation = "concatenate"
                mapi2t_1000, mapt2i_1000, rBX, rBY = valid_fun(hashing_model, args,
                                                test_image, database_image,
                                                test_text, database_text,
                                                test_label, database_label,
                                                separation, task_index, top_k=1000)
            end_time_all = time.time()
            log.info(f'...The all data test is finished...cost time: {(end_time_all - start_time_all)*1000}ms')
            log.info('...test MAP: MAP_1000(i->t): %3.4f, MAP_1000(t->i): %3.4f' % (mapi2t_1000, mapt2i_1000))
            result_dict['I2T'][task_index, args.num_tasks+1] = mapi2t_1000.cpu().numpy()
            result_dict['T2I'][task_index, args.num_tasks+1] = mapt2i_1000.cpu().numpy()
        else:
            start_time_all = time.time()

            test_combined_image = []
            test_combined_text = []
            test_combined_label = []

            database_combined_label = []
            
            for i in range(task_index+1):
                _, _, _, test_image, test_text, test_label, _, _, database_label = get_data(args.dataset_path, i, args.dataset_name)
                test_combined_image.append(test_image)
                test_combined_text.append(test_text)
                test_combined_label.append(test_label)
                database_combined_label.append(database_label)

            test_image = torch.from_numpy(np.concatenate(test_combined_image)).cuda()
            test_text = torch.from_numpy(np.concatenate(test_combined_text)).cuda()
            test_label = torch.from_numpy(np.concatenate(test_combined_label)).cuda()
            database_label = torch.from_numpy(np.concatenate(database_combined_label)).cuda()

            with torch.no_grad():
                separation = "concatenate"
                qBX, qBY = generate_code(hashing_model, test_image, test_text, args, separation, task_index)
                rBX = torch.sign(torch.cat(history_database_code_list_image, dim=0))
                rBY = torch.sign(torch.cat(history_database_code_list_text, dim=0))
                mapi2t_1000 = calc_map_k(qBX, rBY, test_label, database_label, topk=1000)
                mapt2i_1000 = calc_map_k(qBY, rBX, test_label, database_label, topk=1000)
            end_time_all = time.time()
            log.info(f'...The all data test is finished...cost time: {(end_time_all - start_time_all)*1000}ms')
            log.info('...test MAP: MAP_1000(i->t): %3.4f, MAP_1000(t->i): %3.4f' % (mapi2t_1000, mapt2i_1000))
            result_dict['I2T'][task_index, args.num_tasks+1] = mapi2t_1000.cpu().numpy()
            result_dict['T2I'][task_index, args.num_tasks+1] = mapt2i_1000.cpu().numpy()
    if task_index == 0:
        pass
    else:
        if not args.old_dataset_code_is_useful:
            start_time_all = time.time()
            for i in range(task_index+1):
                _, _, _, test_image, test_text, test_label, database_image, database_text, database_label = get_data(args.dataset_path, i, args.dataset_name)
                test_image = test_image.cuda()
                test_text = test_text.cuda()
                test_label = test_label.cuda()
                database_image = database_image.cuda()
                database_text = database_text.cuda()
                database_label = database_label.cuda()

                with torch.no_grad():
                    separation = "single"
                    mapi2t_1000, mapt2i_1000, rBX, rBY = valid_fun(hashing_model, args,
                                                    test_image, database_image,
                                                    test_text, database_text,
                                                    test_label, database_label,
                                                    separation, i, top_k=1000)
                log.info('...The {} data test is finished...'.format(i+1))
                end_time_all = time.time()
                log.info(f'...The divide data test is finished...cost time: {(end_time_all - start_time_all)*1000}ms')
                log.info('...test MAP: MAP_1000(i->t): %3.4f, MAP_1000(t->i): %3.4f' % (mapi2t_1000, mapt2i_1000))
                if task_index == 0:
                    result_dict['I2T'][task_index, args.num_tasks] = mapi2t_1000.cpu().numpy()
                    result_dict['T2I'][task_index, args.num_tasks] = mapt2i_1000.cpu().numpy()
                result_dict['I2T'][task_index, i] = mapi2t_1000.cpu().numpy()
                result_dict['T2I'][task_index, i] = mapt2i_1000.cpu().numpy()
        else:
            for i in range(task_index+1):
                start_time_all = time.time()
                _, _, _, test_image, test_text, test_label, _, _, database_label = get_data(args.dataset_path, i, args.dataset_name)

                test_image = test_image.cuda()
                test_text = test_text.cuda()
                test_label = test_label.cuda()
                database_label = database_label.cuda()

                with torch.no_grad():
                    separation = "single"
                    qBX, qBY = generate_code(hashing_model, test_image, test_text, args, separation, i)
                    rBX = history_database_code_list_image[i]
                    rBY = history_database_code_list_text[i]
                    mapi2t_1000 = calc_map_k(qBX, rBY, test_label, database_label, topk=1000)
                    mapt2i_1000 = calc_map_k(qBY, rBX, test_label, database_label, topk=1000)
                log.info('...The {} data test is finished...'.format(i+1))
                end_time_all = time.time()
                log.info(f'...The divide data test is finished...cost time: {(end_time_all - start_time_all)*1000}ms')
                log.info('...test MAP: MAP_1000(i->t): %3.4f, MAP_1000(t->i): %3.4f' % (mapi2t_1000, mapt2i_1000))
                if task_index == 0:
                    result_dict['I2T'][task_index, args.num_tasks] = mapi2t_1000.cpu().numpy()
                    result_dict['T2I'][task_index, args.num_tasks] = mapt2i_1000.cpu().numpy()
                result_dict['I2T'][task_index, i] = mapi2t_1000.cpu().numpy()
                result_dict['T2I'][task_index, i] = mapt2i_1000.cpu().numpy()


def valid_fun(hashing_model, args, query_x, retrieval_x, query_y, retrieval_y, query_L, retrieval_L, separation, task_index, top_k):
    """
    Generate the hash code and calculate the MAP
    """
    qBX, qBY = generate_code(hashing_model, query_x, query_y, args, separation, task_index)
    rBX, rBY = generate_code(hashing_model, retrieval_x, retrieval_y, args, separation, task_index)

    mapi2t_1000 = calc_map_k(qBX, rBY, query_L, retrieval_L, top_k)
    mapt2i_1000 = calc_map_k(qBY, rBX, query_L, retrieval_L, top_k)
    return mapi2t_1000, mapt2i_1000, rBX, rBY

@torch.no_grad()
def generate_code(hashing_model, X, Y, args, separation, task_index):
    """
    Generate the hash code
    """
    if separation == "single":
        num_data = X.shape[0]
        index = np.linspace(0, num_data - 1, num_data).astype(int)
        B_X = torch.zeros(num_data, args.bit, dtype=torch.float).cuda()
        B_Y = torch.zeros(num_data, args.bit, dtype=torch.float).cuda()

        for i in range(num_data // args.batch_size + 1):
            ind = index[i * args.batch_size: min((i + 1) * args.batch_size, num_data)]
            image = X[ind].type(torch.float).cuda()
            text = Y[ind].type(torch.float).cuda()
            X_hash, Y_hash = hashing_model.test(image, text, task_index)
            B_X[ind] = X_hash
            B_Y[ind] = Y_hash
        B_X = torch.sign(B_X)
        B_Y = torch.sign(B_Y)
        return B_X, B_Y
    elif separation == "concatenate":
        num_data = X.shape[0]
        index = np.linspace(0, num_data - 1, num_data).astype(int)
        B_X = torch.zeros(num_data, args.bit, dtype=torch.float).cuda()
        B_Y = torch.zeros(num_data, args.bit, dtype=torch.float).cuda()

        for i in range(num_data // args.batch_size + 1):
            ind = index[i * args.batch_size: min((i + 1) * args.batch_size, num_data)]
            image = X[ind].type(torch.float).cuda()
            text = Y[ind].type(torch.float).cuda()
            X_hash, Y_hash = hashing_model.cross_task_forward(image, text)
            B_X[ind] = X_hash
            B_Y[ind] = Y_hash
        B_X = torch.sign(B_X)
        B_Y = torch.sign(B_Y)
        return B_X, B_Y
    elif separation == "cross_label":
        num_data = X.shape[0]
        index = np.linspace(0, num_data - 1, num_data).astype(int)
        B_X = torch.zeros(num_data, args.bit, dtype=torch.float).cuda()
        B_Y = torch.zeros(num_data, args.bit, dtype=torch.float).cuda()

        for i in range(num_data // args.batch_size + 1):
            ind = index[i * args.batch_size: min((i + 1) * args.batch_size, num_data)]
            image = X[ind].type(torch.float).cuda()
            text = Y[ind].type(torch.float).cuda()
            X_hash, Y_hash = hashing_model.cross_label_forward(image, text)
            B_X[ind] = X_hash
            B_Y[ind] = Y_hash
        B_X = torch.sign(B_X)
        B_Y = torch.sign(B_Y)
        return B_X, B_Y