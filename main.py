#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Time ： 2024/8/28 21:55
@Auth ： chenxi
@File ：main.py
@IDE ：PyCharm
"""
"""
The entry of the KGEvolve
"""

import argparse
import itertools
import os
import sys
import time
import pickle

import json
import dgl
import numpy as np
import torch
from tqdm import tqdm
import random

sys.path.append("..")
import utils
from utils import build_sub_graph
from haruhi_exp.nets.regcn_net import RecurrentRGCN
from haruhi_exp.nets.gatedgcn_net import GatedGCNNet

from hyperparameter_range import hp_range
import torch.nn.modules.rnn
from collections import defaultdict
from knowledge_graph import _read_triplets_as_list


# os.environ['KMP_DUPLICATE_LIB_OK']='True'


def test(model, history_list, test_list, num_rels, num_nodes, device, all_ans_list, all_ans_r_list, model_name,
         static_graph, mode, pe_init="rw", pe_dim=3):
    """
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param all_ans_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param all_ans_r_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param static_graph
    :param mode
    :return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        checkpoint = torch.load(model_name, map_location=device)
        print("Load Model name: {}. Using best epoch : {}".format(model_name,
                                                                  checkpoint['epoch']))  # use best stat checkpoint
        print("\n" + "-" * 10 + "start testing" + "-" * 10 + "\n")
        model.load_state_dict(checkpoint['state_dict'])

    model.eval()
    # do not have inverse relation in test input
    input_list = [snap for snap in history_list[-args.test_history_len:]]

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        history_glist = [build_sub_graph(num_nodes, num_rels, g, pe_init=pe_init, pe_dim=pe_dim) for g in input_list]
        test_triples_input = torch.LongTensor(test_snap).to(device)
        test_triples_input = test_triples_input.to(device)
        test_triples, final_score, final_r_score = model.predict(history_glist, num_rels, static_graph,
                                                                 test_triples_input)

        mrr_filter_snap_r, mrr_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(test_triples, final_r_score,
                                                                                        all_ans_r_list[time_idx],
                                                                                        eval_bz=1000, rel_predict=1)
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score,
                                                                                all_ans_list[time_idx], eval_bz=1000,
                                                                                rel_predict=0)

        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)

        # relation rank
        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)
        mrr_raw_list_r.append(mrr_snap_r)
        mrr_filter_list_r.append(mrr_filter_snap_r)

        # reconstruct history graph list
        if args.multi_step:
            if not args.relation_evaluation:
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            else:
                predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_list.pop(0)
            input_list.append(test_snap)
        idx += 1

    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent")
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent")
    mrr_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel")
    mrr_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel")
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


from datetime import datetime


def run_experiment(args):
    # load configuration for grid search the best configuration
    # if n_hidden:
    #     args.n_hidden = n_hidden
    # if n_layers:
    #     args.n_layers = n_layers
    # if dropout:
    #     args.dropout = dropout
    # if n_bases:
    #     args.n_bases = n_bases

    # load graph data
    print("loading graph data")
    data = utils.load_data(args["dataset"])
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)

    # model_name = "{}-{}-{}-ly{}-dilate{}-his{}-weight:{}-discount:{}-angle:{}-dp{}|{}|{}|{}-gpu{}" \
    #     .format(args.dataset, args.encoder, args.decoder, args.n_layers, args.dilate_len, args.train_history_len,
    #             args.weight, args.discount, args.angle,
    #             args.dropout, args.input_dropout, args.hidden_dropout, args.feat_dropout, args.gpu)
    format_str = "%Y-%m-%d-%H-%M-%S"
    out_dir = args["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    model_name = f"{datetime.now().strftime(format_str)}"
    # model_name = "ICEWS14s_2024-09-03-21-03-20"
    model_state_file = os.path.join(out_dir, model_name)

    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    # device
    device_id = args["gpu"]["id"]
    device = f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"

    gnn_params = args["gnn"]
    gnn_netparams = args["gnn"]["net_params"]
    gnn_netparams["device"] = device
    decoder_params = args["decoder"]
    # create stat
    if gnn_netparams["add_static_graph"]:
        static_triples = np.array(
            _read_triplets_as_list("./data/" + args["dataset"] + "/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().to(device)
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples,
                                       pe_init=args["pe_init"]["name"], pe_dim=args["pe_init"]["pe_dim"])
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None
    if gnn_params["name"] == "regcn":
        model = RecurrentRGCN(args.decoder,
                              args.encoder,
                              num_nodes,
                              num_rels,
                              num_static_rels,
                              num_words,
                              args.n_hidden,
                              args.opn,
                              sequence_len=args.train_history_len,
                              num_bases=args.n_bases,
                              num_basis=args.n_basis,
                              num_hidden_layers=args.n_layers,
                              dropout=args.dropout,
                              self_loop=args.self_loop,
                              skip_connect=args.skip_connect,
                              layer_norm=args.layer_norm,
                              input_dropout=args.input_dropout,
                              hidden_dropout=args.hidden_dropout,
                              feat_dropout=args.feat_dropout,
                              aggregation=args.aggregation,
                              weight=args.weight,
                              discount=args.discount,
                              angle=args.angle,
                              use_static=args.add_static_graph,
                              entity_prediction=args.entity_prediction,
                              relation_prediction=args.relation_prediction,
                              gpu=args.gpu,
                              analysis=args.run_analysis,
                              pe_init=args.pe_init,
                              pe_dim=args.pe_dim)
    elif args["gnn"]["name"] == "gatedgcn":
        model = GatedGCNNet(decoder_params,
                            num_nodes,
                            num_rels,
                            gnn_netparams,
                            args["params"]["task_weight"],
                            args["pe_init"]["name"],
                            args["pe_init"]["pe_dim"]
                            )



    model = model.to(device)


    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args["params"]["lr"], weight_decay=1e-5)

    if args["params"]["test"] and os.path.exists(model_state_file):
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model,
                                                            train_list + valid_list,
                                                            test_list,
                                                            num_rels,
                                                            num_nodes,
                                                            all_ans_list_test,
                                                            all_ans_list_r_test,
                                                            model_state_file,
                                                            static_graph,
                                                            "test",
                                                            pe_init=args["pe_init"]["name"],
                                                            pe_dim=args["pe_dim"])
    elif args["params"]["test"] and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(
            model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_mrr = 0
        for epoch in range(args["params"]["num_epochs"]):
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            losses_static = []

            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)

            for train_sample_num in tqdm(idx):
                if train_sample_num == 0: continue
                output = train_list[train_sample_num:train_sample_num + 1]
                if train_sample_num - args["params"]["train_history_len"] < 0:
                    input_list = train_list[0: train_sample_num]
                else:
                    input_list = train_list[train_sample_num - args["params"]["train_history_len"]:
                                            train_sample_num]

                # generate history graph
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, device, pe_init=args["pe_init"]["name"], pe_dim=args["pe_init"]["pe_dim"]) for snap in input_list]
                output = [torch.from_numpy(_).long().to(device) for _ in output]
                # get
                loss_e, loss_r, loss_pe = model.get_loss(history_glist, output[0], static_graph)
                loss = args["params"]["task_weight"] * loss_e + (1 - args["task_weight"]) * loss_r + loss_pe

                losses.append(loss.item())
                losses_e.append(loss_e.item())
                losses_r.append(loss_r.item())
                losses_static.append(loss_pe.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args["params"]["grad_norm"])  # clip gradients
                optimizer.step()
                optimizer.zero_grad()

            print(
                "Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), np.mean(losses_static), best_mrr,
                        model_name))

            # validation
            if epoch and epoch % args["params"]["evaluate_every"] == 0:
                mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model,
                                                                    train_list,
                                                                    valid_list,
                                                                    num_rels,
                                                                    num_nodes,
                                                                    all_ans_list_valid,
                                                                    all_ans_list_r_valid,
                                                                    model_state_file,
                                                                    static_graph,
                                                                    mode="train",
                                                                    pe_init=args["pe_init"]["name"],
                                                                    pe_dim=args["pe_init"]["pe_dim"])

                if not args["gnn"]["netparams"]["relation_evaluation"]:  # entity prediction evalution
                    if mrr_raw < best_mrr:
                        if epoch >= args["params"]["num_epochs"]:
                            break
                    else:
                        best_mrr = mrr_raw
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                else:
                    if mrr_raw_r < best_mrr:
                        if epoch >= args["params"]["num_epochs"]:
                            break
                    else:
                        best_mrr = mrr_raw_r
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(model,
                                                            train_list + valid_list,
                                                            test_list,
                                                            num_rels,
                                                            num_nodes,
                                                            all_ans_list_test,
                                                            all_ans_list_r_test,
                                                            model_state_file,
                                                            static_graph,
                                                            mode="test",
                                                            pe_init=args["pe_init"]["name"],
                                                            pe_dim=args["pe_dim"])
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Haruhi')
    parser.add_argument('--config', help="Please give a config.json file", required=True)

    # parser.add_argument("--gpu", type=int, default=-1,
    #                     help="gpu")
    # parser.add_argument("--batch-size", type=int, default=1,
    #                     help="batch-size")
    # parser.add_argument("-d", "--dataset", type=str, required=True,
    #                     help="dataset to use")
    # parser.add_argument("--test", action='store_true', default=False,
    #                     help="load stat from dir and directly test")
    # parser.add_argument("--run-analysis", action='store_true', default=False,
    #                     help="print log info")
    # parser.add_argument("--run-statistic", action='store_true', default=False,
    #                     help="statistic the result")
    # parser.add_argument("--multi-step", action='store_true', default=False,
    #                     help="do multi-steps inference without ground truth")
    # parser.add_argument("--topk", type=int, default=10,
    #                     help="choose top k entities as results when do multi-steps without ground truth")
    # parser.add_argument("--add-static-graph", action='store_true', default=False,
    #                     help="use the info of static graph")
    # parser.add_argument("--add-rel-word", action='store_true', default=False,
    #                     help="use words in relaitons")
    # parser.add_argument("--relation-evaluation", action='store_true', default=False,
    #                     help="save model accordding to the relation evalution")
    #
    # # configuration for encoder RGCN stat
    # parser.add_argument("--weight", type=float, default=1,
    #                     help="weight of static constraint")
    # parser.add_argument("--task-weight", type=float, default=0.7,
    #                     help="weight of entity prediction task")
    # parser.add_argument("--discount", type=float, default=1,
    #                     help="discount of weight of static constraint")
    # parser.add_argument("--angle", type=int, default=10,
    #                     help="evolution speed")
    #
    # parser.add_argument("--encoder", type=str, default="uvrgcn",
    #                     help="method of encoder")
    # parser.add_argument('--local-gnn-type', default="regcn")
    # parser.add_argument("--aggregation", type=str, default="none",
    #                     help="method of aggregation")
    # parser.add_argument("--dropout", type=float, default=0.2,
    #                     help="dropout probability")
    # parser.add_argument("--skip-connect", action='store_true', default=False,
    #                     help="whether to use skip connect in a RGCN Unit")
    # parser.add_argument("--n-hidden", type=int, default=200,
    #                     help="number of hidden units")
    # parser.add_argument("--opn", type=str, default="sub",
    #                     help="opn of compgcn")
    #
    # parser.add_argument("--n-bases", type=int, default=100,
    #                     help="number of weight blocks for each relation")
    # parser.add_argument("--n-basis", type=int, default=100,
    #                     help="number of basis vector for compgcn")
    # parser.add_argument("--n-layers", type=int, default=2,
    #                     help="number of propagation rounds")
    # parser.add_argument("--self-loop", action='store_true', default=True,
    #                     help="perform layer normalization in every layer of gcn ")
    # parser.add_argument("--layer-norm", action='store_true', default=False,
    #                     help="perform layer normalization in every layer of gcn ")
    # parser.add_argument("--relation-prediction", action='store_true', default=False,
    #                     help="add relation prediction loss")
    # parser.add_argument("--entity-prediction", action='store_true', default=False,
    #                     help="add entity prediction loss")
    # parser.add_argument("--split_by_relation", action='store_true', default=False,
    #                     help="do relation prediction")
    #
    # # configuration for stat training
    # parser.add_argument("--n-epochs", type=int, default=500,
    #                     help="number of minimum training epochs on each time step")
    # parser.add_argument("--lr", type=float, default=0.001,
    #                     help="learning rate")
    # parser.add_argument("--grad-norm", type=float, default=1.0,
    #                     help="norm to clip gradient to")
    #
    # # configuration for evaluating
    # parser.add_argument("--evaluate-every", type=int, default=20,
    #                     help="perform evaluation every n epochs")
    #
    # # configuration for decoder
    # parser.add_argument("--decoder", type=str, default="convtranse",
    #                     help="method of decoder")
    # parser.add_argument("--input-dropout", type=float, default=0.2,
    #                     help="input dropout for decoder ")
    # parser.add_argument("--hidden-dropout", type=float, default=0.2,
    #                     help="hidden dropout for decoder")
    # parser.add_argument("--feat-dropout", type=float, default=0.2,
    #                     help="feat dropout for decoder")
    #
    # # configuration for sequences stat
    # parser.add_argument("--train-history-len", type=int, default=10,
    #                     help="history length")
    # parser.add_argument("--test-history-len", type=int, default=20,
    #                     help="history length for test")
    # parser.add_argument("--dilate-len", type=int, default=1,
    #                     help="dilate history graph")
    #
    # # configuration for optimal parameters
    # parser.add_argument("--grid-search", action='store_true', default=False,
    #                     help="perform grid search for best configuration")
    # parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases",
    #                     help="stat to use")
    # parser.add_argument("--num-k", type=int, default=500,
    #                     help="number of triples generated")
    #
    # # configuration for pe
    # parser.add_argument("--pos-init", type=str, default="rw",
    #                     help="init pe")
    # parser.add_argument("--pos-dim", type=int, default=3, help="pos dim")

    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    print(config)
    if config["params"]["grid_search"]:
        out_log = '{}.{}.gs'.format(config["dataset"], config["gnn"]["name"] + "-" + config["decoder"])
        o_f = open(out_log, 'w')
        print("** Grid Search **")
        o_f.write("** Grid Search **\n")
        hyperparameters = config["params"]["tune"].split(',')

        if  config["params"]["tune"] == '' or len(hyperparameters) < 1:
            print("No hyperparameter specified.")
            sys.exit(0)
        grid = hp_range[hyperparameters[0]]
        for hp in hyperparameters[1:]:
            grid = itertools.product(grid, hp_range[hp])  # 生成可迭代对象的笛卡尔积，所有可能的元素组合
        hits_at_1s = {}
        hits_at_10s = {}
        mrrs = {}
        grid = list(grid)
        print('* {} hyperparameter combinations to try'.format(len(grid)))
        o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
        o_f.close()

        for i, grid_entry in enumerate(list(grid)):

            o_f = open(out_log, 'a')

            if not (type(grid_entry) is list or type(grid_entry) is list):
                grid_entry = [grid_entry]
            grid_entry = utils.flatten(grid_entry)
            print('* Hyperparameter Set {}:'.format(i))
            o_f.write('* Hyperparameter Set {}:\n'.format(i))
            signature = ''
            print(grid_entry)
            o_f.write("\t".join([str(_) for _ in grid_entry]) + "\n")
            # def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
            mrr, hits, ranks = run_experiment(config, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3])
            print("MRR (raw): {:.6f}".format(mrr))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr))
            for hit in hits:
                avg_count = torch.mean((ranks <= hit).float())
                print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, avg_count.item()))
    # single run
    else:
        run_experiment(config)
    sys.exit()